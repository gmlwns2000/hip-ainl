import argparse
from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class ArgsType:
    model: Literal['llama32k', 'llama16b', 'qwen'] = 'llama32k'
    job: Literal['ppl', 'mmlu', 'mmmu', 'stream', 'bench_single_layer'] = 'ppl'
    method: Literal['none', 'hip'] = 'hip'
    stride: int = -1
    lora_r: int = 32
    checkpoint: Optional[str] = None
    count: int = 100
    block_size_q: int = 32
    block_size_k: int = 2
    batch_size: int = 1
    k: int = 512
    dense_queries: int = 0
    dense_layers: int = 3

    ensemble: bool = False
    ensemble_model_setting : str = "random_pruning"
    ensemble_method : str = "final_attn"
    ensemble_method_final : str = "query"
    ensemble_method_final_inter_thresh : int = None
    ensemble_method_final_bdd_mask_k : int = 0
    ensemble_timedim_wd : int = 3
    ensemble_per_layer_n : int = 1
    ensemble_per_attn_iter_n : int = 5
    ensemble_model_n : int = 5

    ensemble_particular_layer : int = None
    ensemble_layer_till : int = None
    ensemble_randomness : float = 0.5



def eval_args(
    default_model = 'llama32k',
    default_job = 'ppl',
) -> ArgsType:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=default_model)
    parser.add_argument('--job', type=str, default=default_job)
    parser.add_argument('--method', type=str, default='none')
    parser.add_argument('--stride', type=int, default=-1)
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--count', type=int, default=-1)
    parser.add_argument('--block_size_q', type=int, default=32)
    parser.add_argument('--block_size_k', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--k', type=int, default=512)
    parser.add_argument('--dense_layers', type=int, default=3)
    parser.add_argument('--dense_queries', type=int, default=0)
    parser.add_argument('--name', type=str, default='dev')
    parser.add_argument('--disable_prompt', default=False, action='store_true')
    parser.add_argument('--no_sample', default=False, action='store_true')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--no_quantize', default=False, action='store_true')
    parser.add_argument('--max_tokens', type=int, default=512)

    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--ensemble-model-setting', type=str, default='random_pruning')
    parser.add_argument('--ensemble-method', type=str, default='final_attn')
    parser.add_argument('--ensemble-method-final', type=str, default='query')
    parser.add_argument('--ensemble-method-final-inter-thresh', type=int, default=None) # union
    parser.add_argument('--ensemble-method-final-bdd-mask-k', type=int, default=0)
    parser.add_argument('--ensemble-timedim-wd', type=int, default=3)

    parser.add_argument('--ensemble-per-layer-n', type=int, default=1)
    parser.add_argument('--ensemble-per-attn-iter-n', type=int, default=5)
    parser.add_argument('--ensemble-model-n', type=int, default=20)
    parser.add_argument('--ensemble-particular-layer', type=int, default=None)
    parser.add_argument('--ensemble-layer-till', type=int, default=None)
    parser.add_argument('--ensemble-randomness', type=float, default=5.0)
    parser.add_argument('--ensemble-iter-start-step', type=int, default=1)
    parser.add_argument('--ensemble-iter-n-mode', type=str, default="linear")
    parser.add_argument('--ensemble-iter-n-start', type=int, default=0)
    parser.add_argument('--ensemble-iter-n-factor', type=int, default=2)
    parser.add_argument('--ensemble-iter-n-jump', type=int, default=1)
    parser.add_argument('--ensemble-iter-n-till', type=int, default=None)

    parser.add_argument('--visualize', action='store_true')

    parser.add_argument('--rope_method', type=str, default='none')
    parser.add_argument('--disable_flash', default=False, action='store_true')
    parser.add_argument('--disable_sparq', default=False, action='store_true')
    parser.add_argument('--disable_sliding_window', default=False, action='store_true')
    parser.add_argument('--sampling_method', default='random', type=str)
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--dataset', default='wikitext', type=str)
    args = parser.parse_args()
    print(args)
    return args
