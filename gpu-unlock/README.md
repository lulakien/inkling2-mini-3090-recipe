# gpu-unlock (optional, third party)

[Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) is a kernel module that reprograms GeForce Ampere
cards at boot so FP16/BF16 tensor-core math with **FP32 accumulation** runs at full rate (GeForce firmware
normally runs it at half the FP16-accumulate rate). It does not change results: the same instructions run at
a higher rate.

Why it matters here: prefill is GEMM-bound, and the Humming W4A16 expert kernels and the bf16 attention/dense
GEMMs all use bf16 MMA with fp32 accumulation. Decode is memory-bandwidth bound and is essentially unaffected.
On the earlier Qwen3.8-27B recipe on the same machine the unlock gave +37% prefill, and no decode change.

**All numbers in this repo were measured with the unlock active on both cards.** Without it, expect lower
prefill numbers. Decode, context sizes and quality results do not depend on it.

The two files here are how this machine keeps an unlock boot entry current (CachyOS + Limine +
limine-entry-tool). They are specific to that setup: read them before running anything, and set `TARGETS`
to your own cards' PCI addresses.
- `setup.sh`: one-time install of `/etc/gpu-unlock-entry.conf` and a post hook (needs root). Undo with `--uninstall`.
- `95-gpu-unlock-entry`: the post hook. It recreates a `linux-cachyos-gpu-unlock` boot entry after every
  kernel/initramfs update, with `gpu_unlock.targets=<PCI addresses>` on the kernel command line.

Check it is active: `lsmod | grep gpu_unlock` and `cat /proc/cmdline` (look for `gpu_unlock.targets=`) show that the
module was loaded, not that the cards are still unlocked. The real check is the GEMM rate (`tools/gemm_tflops.py`,
~91 TFLOPS unlocked vs ~64 locked at 280 W). A GPU reset after boot (here: both cards re-initialised by the driver
mid-session, `knvlinkCoreShutdownDeviceLinks` in the kernel log) drops the unlock while the module stays loaded;
with `gpu_unlock.resume=0` only a reboot restored it. Prefill is then ~22% slower, decode unchanged.
Read the upstream README first: the module reprograms the GPU, and you use it at your own risk.
