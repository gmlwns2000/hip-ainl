######### LONG CONTEXT
stride=(8192 12288 16384)

# llama13b_32k dense
# for ((si=0; si<${#stride[@]}; si++)); do
#   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=7 \
#     python -m timber.main.model_eval \
#         --model llama32k \
#         --stride "${stride[si]}" \
#         --method timber \
#         --k 512 \
#         --block_size_q 32 \
#         --block_size_k 2 \
#         --job ppl \
#         --dense_queries 0 \
#         --dense_layers 0 \
#         --overwrite \
#         --count -1
# done
  # llama13b_32k default
#   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=5 \
#   python -m timber.main.model_eval \
#       --model llama32k \
#       --stride "${stride[si]}" \
#       --method timber \
#       --k 512 \
#       --block_size_q 32 \
#       --block_size_k 2 \
#       --job ppl \
#       --dense_queries 0 \
#       --dense_layers 0 \
#       --overwrite \
#       --count -1
# done

# ensemble loop : TODO change thresh hardcoded as 5
# for ((thresh=5; thresh>0; thresh--)); do

t=(5)

for ((ti=0; ti<${#t[@]}; ti+=1)); do
  # for ((layer_till=5; layer_till<26; layer_till+=5)); do
  for ((bdd=0; bdd<=1; bdd++)); do
    for ((si=0; si<${#stride[@]}; si++)); do
      PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=4 \
      python -m timber.main.model_eval \
      --model llama32k \
      --stride "${stride[si]}" \
      --method timber \
      --k 512 \
      --block_size_q 32 \
      --block_size_k 2 \
      --job ppl \
      --dense_queries 0 \
      --ensemble \
      --ensemble-model-setting random_pruning \
      --ensemble-method final_attn \
      --ensemble-method-final query \
      --ensemble-method-final-inter-thresh "${t[ti]}" \
      --ensemble-method-final-bdd-mask-k "${bdd}" \
      --ensemble-layer-till 32 \
      --dense_layers 0 \
      --overwrite \
      --ensemble-model-n 20 \
      --count -1
    done
  done
done

# for ((bdd=0; bdd<2; bdd++)); do
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=5 \
#   python -m timber.main.model_eval \
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
#   --ensemble-method-final-inter-thresh 1 \
#   --ensemble-method-final-bdd-mask-k "${bdd}" \
#   --ensemble-layer-till 32 \
#   --dense_layers 0 \
#   --overwrite \
#   --ensemble-model-n 20 \
#   --count -1
# done

######### LONG CONTEXT

######### THRESH = 1

# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=5 \
#     python -m timber.main.model_eval \
#     --model llama32k \
#     --stride 4096 \
#     --method timber \
#     --k 512 \
#     --block_size_q 32 \
#     --block_size_k 2 \
#     --job ppl \
#     --dense_queries 0 \
#     --ensemble \
#     --ensemble-model-setting random_pruning \
#     --ensemble-method final_attn \
#     --ensemble-method-final query \
#     --ensemble-method-final-inter-thresh 1 \
#     --ensemble-method-final-bdd-mask-k "${bdd}" \
#     --ensemble-layer-till 32 \
#     --dense_layers 0 \
#     --overwrite \
#     --ensemble-model-n 20 \
#     --count -1

# for ((layer_till=25; layer_till>=0; layer_till-=5)); do
#     for ((bdd=0; bdd<2; bdd++)); do
#         PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=5 \
#         python -m timber.main.model_eval \
#         --model llama32k \
#         --stride 4096 \
#         --method timber \
#         --k 512 \
#         --block_size_q 32 \
#         --block_size_k 2 \
#         --job ppl \
#         --dense_queries 0 \
#         --ensemble \
#         --ensemble-model-setting random_pruning \
#         --ensemble-method final_attn \
#         --ensemble-method-final query \
#         --ensemble-method-final-inter-thresh 1 \
#         --ensemble-method-final-bdd-mask-k "${bdd}" \
#         --ensemble-layer-till "${layer_till}" \
#         --dense_layers 0 \
#         --overwrite \
#         --ensemble-model-n 20 \
#         --count -1
#     done
# done