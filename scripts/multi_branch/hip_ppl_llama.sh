#!/bin/bash
stride=(8192 16384 32768 65536 131072)
dlayer=(0 1 2 3)
dataset=('wikitext' 'pg19')

# CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. SA_BLOCK_BK=16 ENFORCE_EAGER=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ATTENTION_BACKEND=HIP_ATTN HIP_DISABLE_AUTOTUNE=1 python hip/main/model_eval.py --method hip --k 512 --block_size_q 64 --block_stride_q 2 --block_size_k 2 --block_stride_k 1 --stride 8192 --dense_layers 3 --no_quantize --overwrite --model llama3.1_8b

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=.
export SA_BLOCK_BK=16
export ENFORCE_EAGER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=HIP_ATTN
export HIP_DISABLE_AUTOTUNE=1

# hip
for ((data=0; data<${#dataset[@]}; data++)); do
    for ((si=0; si<${#stride[@]}; si++)); do
        for ((di=0; di<${#dlayer[@]}; di++)); do
            python hip/main/model_eval.py \
                --method hip \
                --k 512 \
                --block_size_q 64 \
                --block_stride_q 2 \
                --block_size_k 2 \
                --block_stride_k 1 \
                --stride "${stride[si]}" \
                --dense_layers "${dlayer[di]}" \
                --no_quantize \
                --overwrite \
                --model llama3.1_8b \
                --dataset "${dataset[data]}"
        done
    done
done