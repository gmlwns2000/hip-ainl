import torch
import os

def ensemble_random_pruning(
    # ks : torch.Tensor,
    q_hip: torch.Tensor,
    k: torch.Tensor,
    v : torch.Tensor,
    mask_k : int,
    block_size_q : int,
    block_size_k : int,

    ensemble : bool,
    ensemble_model_setting : str,
    ensemble_method : str, 
    ensemble_method_final : str,
    ensemble_method_final_inter_thresh : int,
    ensemble_method_final_bdd_mask_k : int,
    ensemble_per_layer_n : int,
    ensemble_per_attn_iter : bool,
    ensemble_model_n : int,
    ensemble_layer_start : int,
    ensemble_particular_layer : int,
    ensemble_attn_mask_per_layer : torch.Tensor, 

    layer_id : int,
    ):
    
    N_H, TDST, HID = q_hip.shape
    _, TSRC, _ = k.shape
    _N_H, TDST_BQ, MASK_K_BK, MODEL_N = ensemble_attn_mask_per_layer.shape
    assert N_H == _N_H # Not confident abt this
    assert TDST_BQ == TDST//block_size_q
    # indices : [40, 128, 256] = [N*H, TDST//BLOCK_SIZE_Q, mask_k//BLOCK_SIZE_K]
    assert ensemble_method in ['final_attn']
    assert ensemble_method_final in ['query',]

    origin_sparsity = (torch.sum(ensemble_attn_mask_per_layer < TSRC)//ensemble_model_n).item()
    if ensemble_method == "final_attn":
        '''
        [40, 128, 256] * 5
        package in one batch; 
        in batch; output of attentions
        '''
        # ensemble_attn_mask_per_layer [40, 128, 256, 5] to [N*H * TDST//BLOCK_SIZE_Q, mask_k//BLOCK_SIZE_K * ensemble_model_n]
        ensemble_attn_mask_per_layer = ensemble_attn_mask_per_layer.view(_N_H * TDST_BQ, MASK_K_BK * MODEL_N)
        
        unique_ensemble, ensemble_cnt = torch.unique(ensemble_attn_mask_per_layer, return_counts=True)
        
        mask = ensemble_cnt >= ensemble_method_final_inter_thresh
        filtered_unique_ensemble = unique_ensemble[mask] # TODO: use only unique_ensemble?

        if ensemble_method_final_bdd_mask_k == 1:
            ensemble_indices_k_size = mask_k//block_size_k
        else:
            # TODO set to max possible memory : change it more efficiently
            ensemble_indices_k_size = MASK_K_BK * MODEL_N 

        # TODO: Is it better to start plain and concatenate?
        ensembled_indices = torch.full((_N_H*TDST_BQ, ensemble_indices_k_size), 9999999) # change to (N_H, TDST_BQ, ensemble_indices_k_size)

        k_size_max = 0

        # NOTE per_query_token_cnt_diclist is just for analysis
        if os.environ.get('ENSEMBLE_AGREE_DICLIST', '0') == '1':
            per_query_token_cnt_diclist = []

        # NOTE need to compute unique cnt per row - TODO parallel computation
        
        for r in ensemble_attn_mask_per_layer:
            unique_ensemble, ensemble_cnt = torch.unique(r, return_counts=True)
            
            ##### FOR Analysis
            if os.environ.get('ENSEMBLE_AGREE_DICLIST', '0') == '1':
                d = dict(zip(unique_ensemble.tolist(), ensemble_cnt.tolist()))
                sorted_dic = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))
                sorted_dic.pop(9999999, None)
                per_query_token_cnt_diclist.append(sorted_dic)
            #####

            mask = ensemble_cnt >= ensemble_method_final_inter_thresh
            unique_filtered = unique_ensemble[mask]
            counts_filtered = ensemble_cnt[mask]

            sorted_indices = torch.argsort(counts_filtered, descending=True)
            unique_filtered = unique_filtered[sorted_indices]

            len_uf = len(unique_filtered)
            if len_uf > ensemble_indices_k_size:
                unique_filtered = unique_filtered[:ensemble_indices_k_size]

            if len_uf > k_size_max:
                k_size_max = len_uf

            ensembled_indices[:,:len_uf] = unique_filtered

        assert k_size_max <= ensemble_indices_k_size
        assert torch.all(ensembled_indices[:, k_size_max:] == 9999999)
        ensembled_indices = ensembled_indices[:, :k_size_max] # TODO : Is undoing better for padding's perspective?
        ensembled_indices = ensembled_indices.view(_N_H, TDST_BQ, -1)

    k_mask = ensembled_indices != 9999999
    ks = k_mask.sum(dim=2)
    sparsity_per_layer = torch.sum(ensembled_indices!=9999999).item()
    sparsity_ratio = (sparsity_per_layer/origin_sparsity)

    ########
    # print('origin sparsity : ', origin_sparsity)
    # print(f'l_{layer_id} sparsity: ', sparsity_per_layer)
    # print(f'sparsity ratio {(sparsity_per_layer/origin_sparsity)} ')

    # ### saving ensemble result
    # print("PATH: hardcoded to llama 13b chat")
    # if os.environ.get('CHECKOUT_ENSEMBLE', '0') == '1':
    #     os.makedirs(f'./cache/ensemble/llama13b_32k/method/{ensemble_model_setting}_{ensemble_method}_{ensemble_method_final}', exist_ok=True)
    #     torch.save({
    #         'ks' : ks,
    #         'q_hip': q_hip,
    #         'k': k,
    #         'v': v,
    #         'mask_k':mask_k,
    #         'block_size_q':block_size_q,
    #         'block_size_k':block_size_k,
    #         'ensemble': ensemble,
    #         'ensemble_model_setting' : ensemble_model_setting,
    #         'ensemble_method' : ensemble_method,
    #         'ensemble_method_final' : ensemble_method_final,
    #         'ensemble_per_layer_n' : ensemble_per_layer_n,
    #         'ensemble_per_attn_iter' : ensemble_per_attn_iter,
    #         'ensemble_model_n' : ensemble_model_n,
    #         'ensemble_layer_start' : ensemble_layer_start,
    #         'layer_id' : layer_id,

    #         'ensemble_attn_mask_per_layer': ensemble_attn_mask_per_layer,
    #         'per_query_token_cnt_diclist': per_query_token_cnt_diclist,
    #         'ensembled_indices' : ensembled_indices,
    #         'origin_sparsity' : origin_sparsity,
    #         'sparsity_per_layer' : sparsity_per_layer,
    #         'sparse_ratio' : sparsity_ratio,

    #     }, f'./cache/ensemble/llama13b_32k/method/{ensemble_model_setting}_{ensemble_method}_{ensemble_method_final}/l_{layer_id}_m_{ensemble_model_n}_pl_{ensemble_per_layer_n}_pat{ensemble_per_attn_iter}_ln{ensemble_layer_start}.pth')
    #     print(">>> STORED.")
        # input('stored. press enter to continue >>> ')
    ##########
    return ensembled_indices, ks, origin_sparsity, sparsity_per_layer, sparsity_ratio