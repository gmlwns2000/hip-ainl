# llama13b_32k default
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export CUDA_VISIBLE_DEVICES=1
python -m timber.main.model_eval \
  --model llama32k \
  --stride 4096 \
  --method timber \
  --k 512 \
  --block_size_q 32 \
  --block_size_k 2 \
  --job ppl \
  --dense_queries 0 \
  --dense_layers 0 \
  --overwrite \
  --count -1

python -m timber.main.model_eval \
  --model llama32k \
  --stride 4096 \
  --method timber \
  --k 256 \
  --block_size_q 32 \
  --block_size_k 2 \
  --job ppl \
  --dense_queries 0 \
  --dense_layers 0 \
  --overwrite \
  --count -1

python -m timber.main.model_eval \
  --model llama32k \
  --stride 4096 \
  --method timber \
  --k 128 \
  --block_size_q 32 \
  --block_size_k 2 \
  --job ppl \
  --dense_queries 0 \
  --dense_layers 0 \
  --overwrite \
  --count -1

# ensemble loop TODO change thresh hardcoded as 5
# for ((thresh=5; thresh>0; thresh--)); do
# for ((thresh=5; thresh<21; thresh+=5)); do
#   for ((bdd=0; bdd<2; bdd++)); do
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
    # python -m timber.main.model_eval \
    #   --model llama32k \
    #   --stride 4096 \
    #   --method timber \
    #   --k 512 \
    #   --block_size_q 32 \
    #   --block_size_k 2 \
    #   --job ppl \
    #   --dense_queries 0 \
    #   --ensemble \
    #   --ensemble-model-setting random_pruning \
    #   --ensemble-method final_attn \
    #   --ensemble-method-final query \
    #   --ensemble-method-final-inter-thresh "${thresh}" \
    #   --ensemble-method-final-bdd-mask-k "${bdd}" \
    #   --ensemble-layer-till 32 \
    #   --dense_layers 0 \
    #   --overwrite \
    #   --count -1 \
    #   --ensemble-model-n 20 
#   done
# done
