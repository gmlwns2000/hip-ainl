import time
import torch
torch.xpu.is_available()

from hip.models.hip_attention.attention2_draft_prefetch import (
    block_sparse_attention, 
    HiPAttentionArgs, 
    load_checkouts
)

seq_len = 65536
head_dim = 128
num_heads = 32
num_heads_kv = 8
bsz = 1

device = 'xpu'
dtype = torch.bfloat16

def test_index(target_idx):
    q = torch.zeros((1, 1, num_heads, head_dim), device=device, dtype=dtype)
    k = torch.zeros((1, seq_len, num_heads, head_dim), device=device, dtype=dtype)
    v = torch.zeros((1, seq_len, num_heads, head_dim), device=device, dtype=dtype)
    seq_lens = torch.tensor([[seq_len], [seq_len]], dtype=torch.int64, device=device)
    cos = sin = torch.randn((seq_len, head_dim), device=device, dtype=dtype)

    args = HiPAttentionArgs(
        mask_k=2048,
        block_size_q=16,
        block_stride_q=1,
        block_size_k=16,
        block_stride_k=1,
        sink_token_size=128,
        sliding_window_size=1024,
        rope_cos=cos,
        rope_sin=sin,
    )
    
    indices = torch.zeros((bsz * num_heads, 1, 1), device=device, dtype=torch.int32)
    indices.fill_(target_idx // 8 * 8)
    ks = torch.zeros((bsz * num_heads, 1), device=device, dtype=torch.int32)
    ks.fill_(1)
    ks_count = ks.unsqueeze(-1)
    ks_start_end = torch.cat([torch.zeros_like(ks.unsqueeze(-1)), ks.unsqueeze(-1)], dim=-1)
    
    q[:, :, :, :] = (torch.arange(0, head_dim, device=device, dtype=dtype) / head_dim)[None, None, None, :]
    k[:, target_idx:target_idx+1, :, :] = (torch.arange(0, head_dim, device=device, dtype=dtype) / head_dim)[None, None, None, :]
    v[:, :, :, 0] = torch.arange(0, seq_len, device=device, dtype=dtype)[None, :, None] / 65536
    v[:, target_idx, :, -1] = 1

    q = q.expand(bsz, -1, -1, -1)
    k = k.expand(bsz, -1, -1, -1)
    v = v.expand(bsz, -1, -1, -1)
    
    torch.xpu.synchronize()
    start = time.time()

    out = block_sparse_attention(
        q=q, k=k, v=v, 
        seq_lens=seq_lens,
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        args=args, 
        EXTEND_BACKEND='streaming', 
        model_context_length=131072,
    )
    
    # lookup_idx = lookup_canary = 0
    torch.xpu.synchronize()
    end = time.time()
    lookup_idx = out[0, 0, 0, 0].item() * 65536
    lookup_canary = out[0, 0, 0, -1].item()
    print('bsa', (end - start) * 1000, target_idx, lookup_idx, lookup_canary)

    torch.xpu.synchronize()
    start = time.time()
    torch.nn.functional.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3), 
        k.permute(0, 2, 1, 3), 
        v.permute(0, 2, 1, 3), 
        enable_gqa=True,
    )
    torch.xpu.synchronize()
    print('fa2', (time.time()- start) * 1000)

for i in range(0, seq_len, 371):
    test_index(i)