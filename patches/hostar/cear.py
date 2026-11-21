"""TP2 all-reduce for large (prefill) tensors on GPUs without P2P, done by the copy engines so it can
overlap with GEMMs on the SMs ("ceAR").

A rank DMAs (D2H) a piece of its partial result into its slot of a shared, cudaHostRegister'ed /dev/shm
buffer and then bumps its piece counter there with cuStreamWriteValue32 (ordered after the copy on the
send stream). The peer's recv stream waits on that counter with cuStreamWaitValue32 (GPU front end, no SMs)
and DMAs the piece in (H2D). The caller adds mine + peer (fp32 add, one bf16 rounding; a+b == b+a), so both
ranks get the same, NCCL-identical sum.

Protocol: between begin() and end(), both ranks issue the same sequence of send(nbytes at offset) calls,
and recv() the k-th piece only after having sent it themselves (piece k of the peer is identified by the
peer's counter reaching k). Slots are double-buffered by call parity: when a rank sends call k+1 its
compute stream has already waited for all of call k's peer pieces, so the peer had finished call k-1
(including its reads of our parity-(k+1) slot). Eager mode only (prefill shapes are not CUDA-graphed).
Counters are 32-bit and compared with >=; they would wrap after 4e9 pieces."""
import ctypes
import mmap
import os

import torch

_libcuda = ctypes.CDLL("libcuda.so.1")
_wait = _libcuda.cuStreamWaitValue32_v2
_write = _libcuda.cuStreamWriteValue32_v2
for _f in (_wait, _write):
    _f.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
    _f.restype = ctypes.c_int
CU_STREAM_WAIT_VALUE_GEQ = 0x0
FLAG_BYTES = 4096


def shm_size(cap: int) -> int:
    size = FLAG_BYTES + 4 * cap
    return (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE


def create_shm(name: str, cap: int) -> None:
    with open(f"/dev/shm/{name}", "wb") as f:
        f.truncate(shm_size(cap))


class CopyEngineAllReduce:
    def __init__(self, name: str, rank: int, cap: int, device: torch.device):
        self.rank, self.cap, self.device = rank, cap, device
        size = shm_size(cap)
        fd = os.open(f"/dev/shm/{name}", os.O_RDWR)
        self.mm = mmap.mmap(fd, size)
        os.close(fd)
        self.ptr = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
        err = torch.cuda.cudart().cudaHostRegister(self.ptr, size, 2 | 1)  # Mapped | Portable
        assert int(err) == 0, f"cudaHostRegister failed: {err}"
        self.host = torch.frombuffer(self.mm, dtype=torch.uint8)  # registered => pinned for copy_
        self.flag_mine = self.ptr + 128 * rank
        self.flag_peer = self.ptr + 128 * (1 - rank)
        self.send_stream = torch.cuda.Stream(device)
        self.recv_stream = torch.cuda.Stream(device)
        self.tmp = torch.empty(cap, dtype=torch.uint8, device=device)
        self.calls = 0
        self.sent = 0  # pieces sent so far
        self.received = 0  # pieces received so far
        self._keep = []
        self._send_done = []

    # -- low-level protocol -------------------------------------------------------------------
    def begin(self) -> None:
        self.parity = self.calls & 1
        self.calls += 1
        self.recv_stream.wait_stream(torch.cuda.current_stream(self.device))  # tmp free again
        self._send_done.clear()
        self._recv_base = self.received

    def _slot(self, who: int, off: int, nbytes: int) -> torch.Tensor:
        base = FLAG_BYTES + (who * 2 + self.parity) * self.cap + off
        return self.host[base: base + nbytes]

    def send(self, t: torch.Tensor, off: int) -> None:
        """Queue D2H of contiguous t (computed on the current stream) to byte offset `off`."""
        nbytes = t.numel() * t.element_size()
        assert t.is_contiguous() and off + nbytes <= self.cap
        ev = torch.cuda.Event(); ev.record(torch.cuda.current_stream(self.device))
        self.sent += 1
        with torch.cuda.stream(self.send_stream):
            self.send_stream.wait_event(ev)
            self._slot(self.rank, off, nbytes).copy_(t.view(torch.uint8).view(-1), non_blocking=True)
            assert _write(ctypes.c_void_p(self.send_stream.cuda_stream), self.flag_mine,
                          self.sent & 0xFFFFFFFF, 0) == 0
            done = torch.cuda.Event(); done.record(self.send_stream)
        self._send_done.append(done)
        self._keep.append(t)

    def recv(self, like: torch.Tensor, off: int) -> torch.Tensor:
        """Queue H2D of the peer's next piece (same shape/dtype as `like`, at byte offset `off`) and make
        the current stream wait for it. Returns the received tensor (a view of the tmp buffer)."""
        nbytes = like.numel() * like.element_size()
        self.received += 1
        assert self.received <= self.sent
        dst = self.tmp[off: off + nbytes]
        with torch.cuda.stream(self.recv_stream):
            assert _wait(ctypes.c_void_p(self.recv_stream.cuda_stream), self.flag_peer,
                         self.received & 0xFFFFFFFF, CU_STREAM_WAIT_VALUE_GEQ) == 0
            dst.copy_(self._slot(1 - self.rank, off, nbytes), non_blocking=True)
            ev = torch.cuda.Event(); ev.record(self.recv_stream)
        cur = torch.cuda.current_stream(self.device)
        cur.wait_event(ev)
        # this rank's D2H of the same piece must be done before the caller overwrites its source
        cur.wait_event(self._send_done[self.received - self._recv_base - 1])
        return dst.view(like.dtype).view(like.shape)

    def end(self) -> None:
        """Sources of send() must stay intact until sent: the current stream waits for the send stream."""
        torch.cuda.current_stream(self.device).wait_stream(self.send_stream)
        self._keep.clear()
        self._send_done.clear()

    # -- convenience: GEMM in row pieces + all-reduce -----------------------------------------
    def run(self, out: torch.Tensor, pieces):
        """out: [M, N] contiguous buffer. pieces: list of (row_start, row_end, produce) where produce()
        returns this rank's partial result for those rows. Writes the all-reduced sum into out."""
        row_bytes = out.stride(0) * out.element_size()
        self.begin()
        parts = []
        for r0, r1, produce in pieces:
            p = produce().contiguous()
            self.send(p, r0 * row_bytes)
            parts.append((r0, r1, p))
        for r0, r1, p in parts:
            torch.add(p, self.recv(p, r0 * row_bytes), out=out[r0:r1])
        self.end()
        return out
