# llama13b_32k dense
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 \
# python -m timber.main.model_eval \
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
# python -m timber.main.model_eval \
#   --model llama32k \
#   --stride 4096 \
#   --method timber \
#   --k 512 \
#   --block_size_q 32 \
#   --block_size_k 2 \
#   --job ppl \
#   --dense_queries 0 \
#   --dense_layers 0 \
#   --overwrite

# ensemble loop : TODO change thresh hardcoded as 5
# for ((thresh=5; thresh>0; thresh--)); do
# Define an array with layers
layer_till=(5 10 15 20 25)

# Loop through thresholds from 5 to 20 with step 5
for ((thresh = 5; thresh <= 20; thresh += 5)); do
  # Loop through bdd (0 and 1)
  for ((lti = 0; lti < ${#layer_till[@]}; lti++)); do
    for ((bdd = 0; bdd < 2; bdd++)); do
      # Run the Python command with different configurations
      PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=4 \
        python -m timber.main.model_eval \
        --model llama32k \
        --stride 4096 \
        --method timber \
        --k 512 \
        --block_size_q 32 \
        --block_size_k 2 \
        --job ppl \
        --dense_queries 0 \
        --ensemble \
        --ensemble-model-setting random_pruning \
        --ensemble-method final_attn \
        --ensemble-method-final intersection \
        --ensemble-method-final-inter-thresh "${thresh}" \
        --ensemble-method-final-bdd-mask-k "${bdd}" \
        --ensemble-layer-till "${layer_till[lti]}" \
        --dense_layers 0 \
        --overwrite \
        --count -1 \
        --ensemble-model-n 20
    done
  done
done

