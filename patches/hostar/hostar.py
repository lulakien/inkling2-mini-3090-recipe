"""TP2 all-reduce through mapped pinned host memory, for GPUs without P2P (RTX 3090 GeForce).

Each rank's kernel: every CTA copies its chunk of the input into this rank's host slot, fences
(__threadfence_system), publishes a per-CTA sequence number in host memory, spins until the peer's CTA
has published the same sequence number, then reads the peer chunk and writes in + peer (fp32 add,
rounded once; a+b == b+a, so both ranks get bit-identical results). The sequence counters live in
device memory and are bumped by the kernel itself, so the op is CUDA-graph safe.
The host buffer is a /dev/shm file mmapped by both processes and cudaHostRegister'ed (mapped)."""
import ctypes
import mmap
import os

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

// host layout: [flags: 2 ranks x MAXB CTAs x 32 uint32 (128 B each)] [slot(rank, parity): 4 x cap bytes]
// Slots are double-buffered by sequence parity: when a rank starts call k+1 the peer may still be
// reading this rank's slot for call k (it has published call k, so it has finished call k-1).
constexpr int MAXB = 256;

__device__ __forceinline__ unsigned ld_volatile(const unsigned* p) {
  unsigned v; asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v;
}
__device__ __forceinline__ float to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float to_f(__half x) { return __half2float(x); }
template <typename T> __device__ __forceinline__ T from_f(float x);
template <> __device__ __forceinline__ __nv_bfloat16 from_f(float x) { return __float2bfloat16_rn(x); }
template <> __device__ __forceinline__ __half from_f(float x) { return __float2half_rn(x); }

template <typename T>
__global__ void hostar_kernel(const T* inp, T* outp, int64_t n, char* host,
                              int64_t cap, int rank, unsigned* seq) {
  const int b = blockIdx.x, nb = gridDim.x;
  unsigned* flags = reinterpret_cast<unsigned*>(host);
  const int64_t nvec = n * sizeof(T) / 16;          // n * sizeof(T) is a multiple of 16 (checked on host)
  const int64_t per = (nvec + nb - 1) / nb;
  const int64_t lo = b * per, hi = min(nvec, lo + per);
  const uint4* d = reinterpret_cast<const uint4*>(inp);
  uint4* o = reinterpret_cast<uint4*>(outp);
  __shared__ unsigned s;
  if (threadIdx.x == 0) { s = seq[b] + 1; seq[b] = s; }
  __syncthreads();
  const unsigned my = s;
  uint4* mine = reinterpret_cast<uint4*>(host + 2 * MAXB * 128 + (rank * 2 + (my & 1)) * cap);
  const uint4* peer = reinterpret_cast<const uint4*>(host + 2 * MAXB * 128 + ((1 - rank) * 2 + (my & 1)) * cap);
  for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) mine[i] = d[i];
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    volatile unsigned* f = flags + (rank * MAXB + b) * 32;
    *f = my;
    __threadfence_system();
    const unsigned* pf = flags + ((1 - rank) * MAXB + b) * 32;
    while (ld_volatile(pf) - my > 0x7fffffffu) {}  // wait until peer seq >= my (wrap-safe)
  }
  __syncthreads();
  for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    uint4 a = d[i];
    uint4 p;
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(p.x), "=r"(p.y), "=r"(p.z), "=r"(p.w) : "l"(peer + i));
    T* av = reinterpret_cast<T*>(&a);
    const T* pv = reinterpret_cast<const T*>(&p);
#pragma unroll
    for (int j = 0; j < 16 / (int)sizeof(T); ++j) av[j] = from_f<T>(to_f(av[j]) + to_f(pv[j]));
    o[i] = a;
  }
}

// out = inp (this rank) + inp (peer); out may alias inp.
void hostar(torch::Tensor inp, torch::Tensor out, int64_t host_ptr, int64_t cap, int64_t rank,
            torch::Tensor seq, int64_t blocks, int64_t threads) {
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous() && inp.numel() == out.numel());
  TORCH_CHECK(inp.scalar_type() == out.scalar_type() && (inp.numel() * inp.element_size()) % 16 == 0);
  TORCH_CHECK(inp.numel() * inp.element_size() <= cap && blocks <= MAXB && seq.numel() >= blocks);
  const c10::cuda::CUDAGuard guard(inp.device());
  auto stream = c10::cuda::getCurrentCUDAStream(inp.device().index());
  char* host = reinterpret_cast<char*>(host_ptr);
  unsigned* sq = reinterpret_cast<unsigned*>(seq.data_ptr());
  if (inp.scalar_type() == at::kBFloat16)
    hostar_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(inp.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), inp.numel(), host, cap, rank, sq);
  else if (inp.scalar_type() == at::kHalf)
    hostar_kernel<__half><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(inp.data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()), inp.numel(), host, cap, rank, sq);
  else TORCH_CHECK(false, "bf16/fp16 only");
}
// All-gather: out = [rank0 input ; rank1 input] (byte copy, any dtype). Same slot/flag/seq protocol.
__global__ void hostag_kernel(const uint4* d, uint4* o, int64_t nvec, char* host, int64_t cap, int rank,
                              unsigned* seq) {
  const int b = blockIdx.x, nb = gridDim.x;
  unsigned* flags = reinterpret_cast<unsigned*>(host);
  const int64_t per = (nvec + nb - 1) / nb;
  const int64_t lo = b * per, hi = min(nvec, lo + per);
  __shared__ unsigned s;
  if (threadIdx.x == 0) { s = seq[b] + 1; seq[b] = s; }
  __syncthreads();
  const unsigned my = s;
  uint4* mine = reinterpret_cast<uint4*>(host + 2 * MAXB * 128 + (rank * 2 + (my & 1)) * cap);
  const uint4* peer = reinterpret_cast<const uint4*>(host + 2 * MAXB * 128 + ((1 - rank) * 2 + (my & 1)) * cap);
  uint4* own_out = o + rank * nvec;
  uint4* peer_out = o + (1 - rank) * nvec;
  for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) { uint4 v = d[i]; mine[i] = v; own_out[i] = v; }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    volatile unsigned* f = flags + (rank * MAXB + b) * 32;
    *f = my;
    __threadfence_system();
    const unsigned* pf = flags + ((1 - rank) * MAXB + b) * 32;
    while (ld_volatile(pf) - my > 0x7fffffffu) {}
  }
  __syncthreads();
  for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    uint4 p;
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(p.x), "=r"(p.y), "=r"(p.z), "=r"(p.w) : "l"(peer + i));
    peer_out[i] = p;
  }
}

void hostag(torch::Tensor inp, torch::Tensor out, int64_t host_ptr, int64_t cap, int64_t rank,
            torch::Tensor seq, int64_t blocks, int64_t threads) {
  const int64_t nbytes = inp.numel() * inp.element_size();
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous() && out.numel() == 2 * inp.numel());
  TORCH_CHECK(inp.scalar_type() == out.scalar_type() && nbytes % 16 == 0 && nbytes <= cap);
  TORCH_CHECK(blocks <= MAXB && seq.numel() >= blocks);
  const c10::cuda::CUDAGuard guard(inp.device());
  auto stream = c10::cuda::getCurrentCUDAStream(inp.device().index());
  hostag_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const uint4*>(inp.data_ptr()),
      reinterpret_cast<uint4*>(out.data_ptr()), nbytes / 16, reinterpret_cast<char*>(host_ptr), cap, rank,
      reinterpret_cast<unsigned*>(seq.data_ptr()));
}
"""
CPP_SRC = ("void hostar(torch::Tensor inp, torch::Tensor out, int64_t host_ptr, int64_t cap, int64_t rank, "
           "torch::Tensor seq, int64_t blocks, int64_t threads);\n"
           "void hostag(torch::Tensor inp, torch::Tensor out, int64_t host_ptr, int64_t cap, int64_t rank, "
           "torch::Tensor seq, int64_t blocks, int64_t threads);")

MAXB = 256
FLAG_BYTES = 2 * MAXB * 128
_ext = None


def ext():
    """Build (first time, ~1 min) or load the extension from <this dir>/build."""
    global _ext
    if _ext is None:
        build_dir = os.environ.get("HOSTAR_BUILD_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
        os.makedirs(build_dir, exist_ok=True)
        _ext = load_inline("hostar_ag_ext", CPP_SRC, cuda_sources=CUDA_SRC, functions=["hostar", "hostag"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_86,code=sm_86"],
                           build_directory=build_dir)
    return _ext


def shm_size(cap: int) -> int:
    size = FLAG_BYTES + 4 * cap
    return (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE


def create_shm(name: str, cap: int) -> None:
    """Create (zeroed) the shared host buffer; call once, before either rank constructs HostAllReduce."""
    with open(f"/dev/shm/{name}", "wb") as f:
        f.truncate(shm_size(cap))


class HostAllReduce:
    """Both ranks construct this with the same shm name, after create_shm()."""

    def __init__(self, name: str, rank: int, cap: int, device: torch.device, blocks: int = 4,
                 threads: int = 256):
        self.rank, self.cap, self.blocks, self.threads = rank, cap, blocks, threads
        size = shm_size(cap)
        fd = os.open(f"/dev/shm/{name}", os.O_RDWR)
        self.mm = mmap.mmap(fd, size)
        os.close(fd)
        self.ptr = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
        rt = torch.cuda.cudart()
        err = rt.cudaHostRegister(self.ptr, size, 2 | 1)  # cudaHostRegisterMapped | Portable
        assert int(err) == 0, f"cudaHostRegister failed: {err}"
        # with UVA (64-bit Linux) a registered host address is also its device address
        self.dptr = self.ptr
        self.seq = torch.zeros(MAXB, dtype=torch.int32, device=device)
        ext()

    def eligible(self, x: torch.Tensor) -> bool:
        nbytes = x.numel() * x.element_size()
        return (x.is_cuda and x.is_contiguous() and x.dtype in (torch.bfloat16, torch.float16)
                and nbytes % 16 == 0 and 0 < nbytes <= self.cap)

    def all_reduce(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(x)
        ext().hostar(x, out, self.dptr, self.cap, self.rank, self.seq, self.blocks, self.threads)
        return out

    def eligible_gather(self, x: torch.Tensor) -> bool:
        nbytes = x.numel() * x.element_size()
        return x.is_cuda and x.is_contiguous() and nbytes % 16 == 0 and 0 < nbytes <= self.cap

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [rank0 x; rank1 x] stacked along dim 0 (NCCL all_gather layout)."""
        out = torch.empty((2 * x.shape[0],) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
        ext().hostag(x, out, self.dptr, self.cap, self.rank, self.seq, self.blocks, self.threads)
        return out
