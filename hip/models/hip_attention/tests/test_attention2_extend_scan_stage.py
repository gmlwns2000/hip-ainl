import time
import torch
print(torch.xpu.is_available(), torch.xpu.device_count())

from hip.models.hip_attention.attention2_draft_sampling_extend import (
    HiPAttentionArgs,
    ScanStage,
    dual_stage_quadratic_hip_attention
)

seq_len = 16384
head_dim = 128
num_heads = 32
num_heads_kv = 8
bsz = 1

device = 'xpu'
dtype = torch.bfloat16

print('start allocate')

q = torch.zeros((1, 1, num_heads, head_dim), dtype=dtype).to(device)
k = torch.zeros((1, seq_len, num_heads_kv, head_dim), dtype=dtype).to(device)
v = torch.zeros((1, seq_len, num_heads_kv, head_dim), dtype=dtype).to(device)
seq_lens = torch.tensor([[seq_len], ] * bsz, dtype=torch.int64).to(device)
cos = sin = torch.randn((seq_len, head_dim), dtype=dtype).to(device)

q = q.expand(bsz, -1, -1, -1)
k = k.expand(bsz, -1, -1, -1)
v = v.expand(bsz, -1, -1, -1)

for i in range(10):
    print('start')
    torch.xpu.synchronize()
    start = time.time()

    out = dual_stage_quadratic_hip_attention(
        q=q, k=k, v=v,
        args=HiPAttentionArgs(
            block_size_k=16,
            block_stride_k=1,
            sink_token_size=128,
            sliding_window_size=1024,
            rope_cos=cos,
            rope_sin=sin,
        ),
        second_stage_k=2048,
        stages=[
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=4,
                stage_chunk_size=256,
                stage_k=None,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=4,
                stage_chunk_size=32,
                stage_k=8192,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=16,
                stage_block_stride_q=1,
                stage_chunk_size=8,
                stage_k=4096,
                stage_stride=1,
            ),
        ],
        block_sparse_block_size_q=16,
        model_context_length=131072,
        sa_extend_backend='streaming',
    )

    torch.xpu.synchronize()
    print('took', (time.time() - start) * 1000, 'ms')