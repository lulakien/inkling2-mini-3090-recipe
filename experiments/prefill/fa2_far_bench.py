"""FA2 (vLLM's sm_80 build) on the 'far prefix' of a deep prefill chunk: 2048 queries against L keys, no causal mask,
12 query heads / 4 KV heads per GPU (TP2), head dim 128, bf16. Prints ms and TFLOPS; contiguous and paged (64-token pages)."""
import torch, sys
from vllm.vllm_flash_attn import flash_attn_varlen_func
torch.manual_seed(0); dev = "cuda"
Q, H, HK, D = 2048, 12, 4, 128
for L in (131072, 262144, 480000):
    q = torch.randn(Q, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(L, HK, D, device=dev, dtype=torch.bfloat16); v = torch.randn_like(k)
    cq = torch.tensor([0, Q], device=dev, dtype=torch.int32); ck = torch.tensor([0, L], device=dev, dtype=torch.int32)
    def run_contig():
        return flash_attn_varlen_func(q, k, v, max_seqlen_q=Q, cu_seqlens_q=cq, max_seqlen_k=L, cu_seqlens_k=ck,
                                      softmax_scale=D ** -0.5, causal=False, return_softmax_lse=True, fa_version=2)
    B = 64; nb = (L + B - 1) // B
    kc = torch.zeros(nb, B, HK, D, device=dev, dtype=torch.bfloat16); vc = torch.zeros_like(kc)
    kc.view(-1, HK, D)[:L] = k; vc.view(-1, HK, D)[:L] = v
    bt = torch.arange(nb, device=dev, dtype=torch.int32)[None]
    su = torch.tensor([L], device=dev, dtype=torch.int32)
    def run_paged():
        return flash_attn_varlen_func(q, kc, vc, max_seqlen_q=Q, cu_seqlens_q=cq, max_seqlen_k=L, seqused_k=su,
                                      block_table=bt, softmax_scale=D ** -0.5, causal=False, return_softmax_lse=True,
                                      fa_version=2)
    flops = 4 * Q * L * H * D
    for name, fn in (("contiguous", run_contig), ("paged64", run_paged)):
        try:
            for _ in range(2): fn()
            torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
            for _ in range(5): fn()
            e.record(); torch.cuda.synchronize(); ms = s.elapsed_time(e) / 5
            print(f"L={L:>7} {name:>10}: {ms:7.2f} ms  {flops / ms / 1e9:6.1f} TFLOPS", flush=True)
        except Exception as ex:
            print(f"L={L:>7} {name:>10}: FAILED {type(ex).__name__}: {str(ex)[:200]}", flush=True)
    del q, k, v, kc, vc; torch.cuda.empty_cache()
