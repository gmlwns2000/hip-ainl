# llama13b_32k dense
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 \
# python -m hip.main.model_eval \
  #   --model llama32k \
  #   --stride 4096 \
  #   --method none \
  #   --k 512 \
  #   --block_size_q 32 \
  #   --block_size_k 2 \
  #   --job ppl \
  #   --dense_queries 0 \
  #   --dense_layers 0 \
  #   --overwrite

# # llama13b_32k default
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 \
# python -m hip.main.model_eval \
  #   --model llama32k \
  #   --stride 4096 \
  #   --method hip \
  #   --k 512 \
  #   --block_size_q 32 \
  #   --block_size_k 2 \
  #   --job ppl \
  #   --dense_queries 0 \
  #   --dense_layers 0 \
  #   --overwrite

# ensemble loop : TODO change thresh hardcoded as 5
# for ((thresh=5; thresh>0; thresh--)); do
for ((thresh=5; thresh<10; thresh+=5)); do
  for ((layer_till=5; layer_till<11; layer_till+=5)); do
    for ((bdd=0; bdd<2; bdd++)); do
        PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=5 \
        python -m hip.main.model_eval \
        --model llama32k \
        --stride 4096 \
        --method hip \
        --k 512 \
        --block_size_q 32 \
        --block_size_k 2 \
        --job ppl \
        --dense_queries 0 \
        --ensemble \
        --ensemble-model-setting random_pruning \
        --ensemble-method final_attn \
        --ensemble-method-final query \
        --ensemble-method-final-inter-thresh "${thresh}" \
        --ensemble-method-final-bdd-mask-k "${bdd}" \
        --ensemble-layer-till "${layer_till}" \
        --dense_layers 0 \
        --overwrite \
        --ensemble-model-n 20 \
        --count -1
    done
  done
done
