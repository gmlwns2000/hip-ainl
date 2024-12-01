import time
import torch
import argparse
import subprocess
import json
import os
import matplotlib.pyplot as plt

print(torch.xpu.is_available(), torch.xpu.device_count())

from hip.models.hip_attention.attention2_draft_sampling_extend import (
    HiPAttentionArgs,
    ScanStage,
    dual_stage_quadratic_hip_attention
)

def exp(args):
    seq_len = args.tsrc
    head_dim = 128
    num_heads = 32
    num_heads_kv = 8
    bsz = 1

    device = 'xpu'
    dtype = torch.bfloat16

    print('start allocate')

    q = torch.zeros((1, args.tdst, num_heads, head_dim), dtype=dtype).to(device)
    k = torch.zeros((1, seq_len, num_heads_kv, head_dim), dtype=dtype).to(device)
    v = torch.zeros((1, seq_len, num_heads_kv, head_dim), dtype=dtype).to(device)
    seq_lens = torch.tensor([[seq_len], ] * bsz, dtype=torch.int64).to(device)
    cos = sin = torch.randn((seq_len, head_dim), dtype=dtype).to(device)
    is_decode = q.shape[1] == 1

    q = q.expand(bsz, -1, -1, -1)
    k = k.expand(bsz, -1, -1, -1)
    v = v.expand(bsz, -1, -1, -1)

    torch.xpu.synchronize()

    previous_metadata = None
    samples = []
    samples_gpu_power = []
    samples_cpu_power = []
    samples_gpu_clock = []
    for i in range(args.n):
        start = time.time()
        if args.method == 'fa2':
            torch.nn.functional.scaled_dot_product_attention(
                q.permute(0, 2, 1, 3), 
                k.permute(0, 2, 1, 3), 
                v.permute(0, 2, 1, 3), 
                enable_gqa=True,
            )
        elif args.method == 'hip':
            out, metadata = dual_stage_quadratic_hip_attention(
                q=q, k=k, v=v,
                args=HiPAttentionArgs(
                    block_size_k=16,
                    block_stride_k=1,
                    sink_token_size=64,
                    sliding_window_size=256,
                    rope_cos=cos,
                    rope_sin=sin,
                ),
                second_stage_k=1024,
                stages=[
                    ScanStage(
                        stage_block_size_q=16,
                        stage_block_stride_q=1,
                        stage_chunk_size=256,
                        stage_k=None,
                        stage_stride=1,
                    ),
                    ScanStage(
                        stage_block_size_q=16,
                        stage_block_stride_q=1,
                        stage_chunk_size=32,
                        stage_k=32768,
                        stage_stride=1,
                    ),
                    ScanStage(
                        stage_block_size_q=16,
                        stage_block_stride_q=1,
                        stage_chunk_size=16,
                        stage_k=8192,
                        stage_stride=1,
                    ),
                ] if is_decode else [
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
                        stage_k=32768,
                        stage_stride=1,
                    ),
                    ScanStage(
                        stage_block_size_q=64,
                        stage_block_stride_q=4,
                        stage_chunk_size=16,
                        stage_k=8192,
                        stage_stride=1,
                    ),
                ],
                block_sparse_block_size_q=16,
                model_context_length=131072,
                sa_extend_backend='streaming',
                cached_metadata=previous_metadata,
            )
            previous_metadata = metadata
            if ((i + 1) % (8 if is_decode else 1)) == 0:
                previous_metadata = None
        else:
            raise Exception(args.method)
        
        handle = subprocess.Popen('intel_gpu_top -J -s 10'.split(), stdout=subprocess.PIPE)
        torch.xpu.synchronize()
        elapsed = (time.time() - start) * 1000

        handle.kill()
        clock = handle.stdout.read().decode('utf-8')
        # print(clock)

        clock = f'[{clock}]'
        clock = json.loads(clock)
        device_measures = clock
        # print(sample)
        
        clock = 0
        gpu_power = 0
        package_power = 0
        count = 0
        for m in device_measures:
            iclock = m['frequency']['actual']
            igpu_power = m['power']['GPU']
            ipackage_power = m['power']['Package']
            if igpu_power > 0:
                clock += iclock
                gpu_power += igpu_power
                package_power += ipackage_power
                count += 1
        clock /= count
        gpu_power /= count
        package_power /= count

        if i > 3:
            samples.append(elapsed)
            samples_gpu_clock.append(clock)
            samples_gpu_power.append(gpu_power)
            samples_cpu_power.append(package_power)
        print(f'[{i}] {args.method} took {elapsed:.2f} ms, {clock:.3f} mhz, {gpu_power:.3f} GPU W, {package_power:.3f} CPU W')
    
    os.makedirs('./saves/test_xpu_gen3', exist_ok=True)

    plt.clf()
    plt.scatter(x=samples_gpu_clock, y=samples)
    plt.title(f'[{args.method}] Q:{tuple(q.shape)}, KV:{tuple(k.shape)}')
    plt.xlabel('GPU clock (Mhz)')
    plt.ylabel('Latency (ms)')
    plt.grid()
    plt.savefig(f'./saves/test_xpu_gen3/{args.method}_clock.png', dpi=300)

    plt.clf()
    plt.scatter(x=samples_gpu_power, y=samples)
    plt.title(f'[{args.method}] Q:{tuple(q.shape)}, KV:{tuple(k.shape)}')
    plt.xlabel('GPU Power (W)')
    plt.ylabel('Latency (ms)')
    plt.grid()
    plt.savefig(f'./saves/test_xpu_gen3/{args.method}_gpu_power.png', dpi=300)

    plt.clf()
    plt.scatter(x=samples_cpu_power, y=samples)
    plt.title(f'[{args.method}] Q:{tuple(q.shape)}, KV:{tuple(k.shape)}')
    plt.xlabel('Package Power (W)')
    plt.ylabel('Latency (ms)')
    plt.grid()
    plt.savefig(f'./saves/test_xpu_gen3/{args.method}_cpu_power.png', dpi=300)

    result = {
        'latency': samples,
        'clock': samples_gpu_clock,
        'gpu_power': samples_gpu_power,
        'cpu_power': samples_cpu_power,
    }
    with open(f'./saves/test_xpu_gen3/{args.method}_result.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f'avg {sum(samples) / len(samples):.1f} ms')

def plot():
    results = {}
    with open('./saves/test_xpu_gen3/fa2_result.json', 'r') as f:
        results['fa2'] = json.load(f)
    with open('./saves/test_xpu_gen3/hip_result.json', 'r') as f:
        results['hip'] = json.load(f)
    
    def render(target, y_metric='latency'):
        plt.clf()
        for method in results.keys():
            plt.scatter(x=results[method][target], y=results[method][y_metric], label=method)
        plt.grid()
        plt.xlabel(target)
        plt.ylabel('latency')
        plt.legend()
        plt.savefig(f'./saves/test_xpu_gen3/summary_{target}.png', dpi=300, bbox_inches='tight')

        plt.clf()
        for method in results.keys():
            plt.plot(list(range(len(results[method][target]))), results[method][target], label=method)
        plt.grid()
        plt.xlabel('time')
        plt.ylabel(target)
        plt.legend()
        plt.savefig(f'./saves/test_xpu_gen3/summary_{target}_time.png', dpi=300, bbox_inches='tight')
    
    for metric in ['latency', 'clock', 'gpu_power', 'cpu_power']:
        render(metric)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='hip')
    parser.add_argument('--tsrc', default=131072, type=int)
    parser.add_argument('--tdst', default=1, type=int)
    parser.add_argument('--n', default=100, type=int)
    args = parser.parse_args()

    if args.method in ['fa2', 'hip']:
        exp(args)
    elif args.method == 'plot':
        plot()
    else:
        pass

if __name__ == '__main__':
    main()