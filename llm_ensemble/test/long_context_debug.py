from ...timber.models.timber_attention.attention1_block_gpu import timber_attention
import torch

t = torch.load(f'./cache/stride_debug/s16384.pth', map_location=torch.device('cpu'))

attn_output_timber, (indices, ks, attn_probs, sparsity, ensemble_cnt_filtered_or_none) = timber_attention(
    q = t['q'],
    k = t['k'],
    v = t['v'],

    attention_mask = t['attention_mask'],
    w_start = t['w_start'],
    n_patches=t['n_patches'],
    mask_k=t['mask_k'],
    scale_up = t['scale_up'],
    is_causal=t['is_causal'],

    block_size_q=t['block_size_q'],
    block_size_k=t['block_size_k'],
    reduce_method=t['reduce_method'],
    reduce_stride=t['reduce_stride'],
    
    chunking=t['chunking'],
    chunk_size=t['chunk_size'],

    is_flash=t['is_flash'],
    enable_sparq=t['enable_sparq'],
    
    sampling_metho=t['sampling_method'],

    ensemble=t['ensemble'],
    ensemble_model_setting=t['ensemble_model_setting'],
    ensemble_method=t['ensemble_method'],
    ensemble_method_final=t['ensemble_method_final'],
    ensemble_method_final_inter_thresh=t['ensemble_method_final_inter_thresh'],
    ensemble_method_final_bdd_mask_k=t['ensemble_method_final_bdd_mask_k'],
    ensemble_method_final_timedim=t['ensemble_method_final_timedim'],
    ensemble_per_layer_n=t['ensemble_per_layer_n'],
    ensemble_per_attn_iter_n=t['ensemble_per_attn_iter_n'],
    ensemble_model_n=t['ensemble_model_n'],
    ensemble_particular_layer=t['ensemble_particular_layer'],
    ensemble_layer_till=t['ensemble_layer_till'],
    ensemble_randomness=t['ensemble_randomness'],

    layer_id=t['layer_id'],    
    using_sliding_window=t['using_sliding_window'],
    sliding_window_size=t['sliding_window_size'],
    
    dense_queries_exp=t['dense_queries_exp'],
    
    rope_method=t['rope_method'],
    rope_cos=t['rope_cos'],
    rope_sin=t['rope_sin'],
    position_ids=t['position_ids'],
    
    self_extend_scale=t['self_extend_scale'],
    self_extend_window=t['self_extend_window',]
    
    using_precomputed_mask=t['using_precomputed_mask'],
    precomputed_indices=t['precomputed_indices'],
    precomputed_ks=t['precomputed_ks'],
)

