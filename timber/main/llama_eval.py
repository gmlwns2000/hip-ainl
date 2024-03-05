import os
import time
import traceback
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm
import argparse
from transformers import TextStreamer
import pathlib

from vllm import LLM
from vllm.model_executor.layers.attention import PagedAttention
from peft import LoraConfig, TaskType, PeftModel
from peft import get_peft_model, prepare_model_for_kbit_training
from timber.models.modeling_llama import LlamaForCausalLM, LlamaConfig
from timber.utils import seed, get_bench

from timber.main.jobs.bench_single_layer import job_bench_single_layer
from timber.main.jobs.ppl import job_ppl
from timber.main.jobs.stream import job_stream
from timber.main.jobs.mmlu import job_mmlu
from timber.main.jobs.booksum import job_booksum
from timber.main.jobs.merge_lora import job_merge_lora
from timber.main.eval_args import eval_args, ArgsType

def load_vllm_model(args: ArgsType):
    from vllm import LLM
    
    device = 'cuda:0'
    MODELS = {
        'vllm_llama32k': 'togethercomputer/LLaMA-2-7B-32K',
        'vllm_llama100k': 'Yukang/Llama-2-7b-longlora-100k-ft',
        'vllm_llama32k_instruct': 'togethercomputer/Llama-2-7B-32K-Instruct',
        'vllm_llama1b': 'princeton-nlp/Sheared-LLaMA-1.3B',
        'vllm_llama7b': 'meta-llama/Llama-2-7b-hf',
        'vllm_llama13b': 'meta-llama/Llama-2-13b-hf',
        'vllm_qwen14b': 'Qwen/Qwen1.5-14B-Chat-GPTQ-Int4',
        'vllm_qwen14b_local': './Qwen1.5-14B-Chat-GPTQ-Int4',
        'vllm_qwen14b_int8_local': './Qwen1.5-14B-Chat-GPTQ-Int8',
        'vllm_qwen14b_noquant_local': './Qwen1.5-14B-Chat',
        'vllm_qwen7b': 'Qwen/Qwen1.5-7B-Chat-GPTQ-Int4',
        'vllm_qwen0.5b': 'Qwen/Qwen1.5-0.5B-Chat',
        'vllm_pythia70m': 'EleutherAI/pythia-70m',
        'vllm_yi6b': '01-ai/Yi-6B-200K',
        'vllm_yi34b': 'brucethemoose/Yi-34B-200K-RPMerge',
    }
    if args.model in MODELS:
        model_id = MODELS[args.model]
    else:
        model_id = args.model.replace('vllm_', '')
    print(f'Loading model {model_id}')
    
    assert args.checkpoint is None
    
    seq_len = args.stride
    # seq_len = 10600
    model = LLM(
        model_id,
        max_context_len_to_capture=seq_len,
        max_num_seqs=args.batch_size,
        max_model_len=seq_len,
        swap_space=0,
        kv_cache_dtype='fp8_e5m2',
        dtype='half',
        gpu_memory_utilization=0.85,
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=os.environ.get('FORCE_EAGER','0')=='1',
        trust_remote_code=True,
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer, device

def load_model(args):
    if args.model.startswith('vllm'):
        return load_vllm_model(args)
    
    device = 'cuda:0'
    MODELS = {
        'llama32k': 'togethercomputer/LLaMA-2-7B-32K',
        'llama13b': 'meta-llama/Llama-2-13b-hf',
        'qwen14b': 'Qwen/Qwen1.5-14B-Chat',
        'qwen7b': 'Qwen/Qwen1.5-7B-Chat',
        'qwen0.5b': 'Qwen/Qwen1.5-0.5B-Chat',
    }
    if args.model in MODELS:
        model_id = MODELS[args.model]
    else:
        model_id = args.model
    print(f'Loading model {model_id}')

    config = LlamaConfig.from_pretrained(model_id)
    config._attn_implementation = config.attn_implementation = 'sdpa'
    
    infer_dtype = torch.bfloat16
    # infer_dtype = torch.float32

    ModelClass = LlamaForCausalLM
    if args.method == 'none':
        ModelClass = transformers.LlamaForCausalLM

    model = ModelClass.from_pretrained(
        model_id,
        config=config, 
        #load_in_4bit=True,
        device_map="auto",
        quantization_config=transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_skip_modules=['tree_avgpool_scaler'],
            bnb_4bit_compute_dtype=infer_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ) if not args.no_quantize else None,
        torch_dtype=infer_dtype,
        trust_remote_code=True,
    )
    
    for m in model.modules():
        if hasattr(m, 'attention_method'):
            m.attention_method = args.method
            m.tree_k = args.k
            m.tree_block_size_q = args.block_size_q
            m.tree_block_size_k = args.block_size_k
            m.tree_using_context_avg = True
            m.tree_dense_queries = args.dense_queries
    
    if args.method != 'none' and args.checkpoint is not None:
        if pathlib.Path(args.checkpoint).is_dir():
            # is peft checkpoint
            # Load peft pretrained
            print(f"Loading peft model from {args.checkpoint}")
            model = PeftModel.from_pretrained(model, args.checkpoint)

        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=True,
                r=args.lora_r,
                lora_alpha=args.lora_r//2,
                lora_dropout=0.0,
                target_modules=[
                    'q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj',
                    # 'input_layernorm', 'post_attention_layernorm'
                ],
                modules_to_save=[
                    'tree_avgpool_scaler',
                    'input_layernorm', 'post_attention_layernorm'
                ]
            )

            model = get_peft_model(model, peft_config)

            state_dict = torch.load(args.checkpoint, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            keys = list(state_dict.keys())
            for key in keys:
                x = state_dict[key]
                state_dict[key.strip('model.')] = x
                del state_dict[key]
            try:
                result = model.load_state_dict(state_dict, strict=False)
                print('load result', result)
            except RuntimeError as e:
                pass

        model = model.to(infer_dtype)
        print('lora checkpoint loaded from', args.checkpoint)

    elif args.method != 'none':
        for m in model.modules():
            if hasattr(m, 'attention_method'):
                m.tree_using_context_avg = False
    
    model = model.eval()
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    
    return model, tokenizer, device

def main():
    seed()
    
    args = eval_args()
    
    assert args.job in ['ppl', 'stream', 'mmlu', 'bench_single_layer', 'booksum', 'merge_lora']
    
    model, tokenizer, device = load_model(args)

    if args.job == 'ppl':
        job_ppl(args, model, tokenizer, device)
    elif args.job == 'stream':
        job_stream(args, model, tokenizer, device)
    elif args.job == 'mmlu':
        job_mmlu(args, model, tokenizer, device)
    elif args.job == 'bench_single_layer':
        job_bench_single_layer(args, model, tokenizer, device)
    elif args.job == 'booksum':
        job_booksum(args, model, tokenizer, device)
    elif args.job == 'merge_lora':
        job_merge_lora(args, model, tokenizer, device)
    else:
        raise Exception()

if __name__ == '__main__':
    main()