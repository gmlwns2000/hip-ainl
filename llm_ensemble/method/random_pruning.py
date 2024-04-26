import math
from timber.trainer.timber_trainer import load_model
import torch
import os

def ensemble_random_pruning(
    # ks : torch.Tensor,
    q_timber: torch.Tensor,
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
    ensemble_per_attn_iter_n : int,
    ensemble_model_n : int,
    ensemble_particular_layer : int,
    ensemble_attn_mask_per_layer : torch.Tensor, 
    ensemble_randomness : float,

    layer_id : int,
    ):
    N_H, TDST, HID = q_timber.shape
    _, TSRC, _ = k.shape
    _N_H, TDST_BQ, MASK_K_BK, MODEL_N = ensemble_attn_mask_per_layer.shape
    assert N_H == _N_H # Not confident abt this
    assert TDST_BQ == TDST//block_size_q
    # indices : [40, 128, 256] = [N*H, TDST//BLOCK_SIZE_Q, mask_k//BLOCK_SIZE_K]
    assert ensemble_method in ['final_attn']
    assert ensemble_method_final in ['intersection',]

    origin_sparsity = (torch.sum(ensemble_attn_mask_per_layer < TSRC)//ensemble_model_n).item()
    if ensemble_method == "final_attn":
        '''
        [40, 128, 256] * 5
        package in one batch; 
        in batch; output of attentions
        '''
        # ensemble_attn_mask_per_layer [40, 128, 256, 5] to [N*H * TDST//BLOCK_SIZE_Q, mask_k//BLOCK_SIZE_K * ensemble_model_n]
        # ensemble_attn_mask_per_layer = ensemble_attn_mask_per_layer.view(_N_H * TDST_BQ, MASK_K_BK * MODEL_N)
        
        # unique_ensemble, ensemble_cnt = torch.unique(ensemble_attn_mask_per_layer, return_counts=True)
        
        # mask = ensemble_cnt >= ensemble_method_final_inter_thresh
        # filtered_unique_ensemble = unique_ensemble[mask] # TODO: use only unique_ensemble?

        if ensemble_method_final_bdd_mask_k == 1:
            ensemble_indices_k_size = mask_k//block_size_k
        else:
            # TODO set to max possible memory : change it more efficiently
            ensemble_indices_k_size = MASK_K_BK * MODEL_N 

        # TODO: Is it better to start plain and concatenate?
        # ensembled_indices = torch.full((_N_H*TDST_BQ, ensemble_indices_k_size), 9999999) # change to (N_H, TDST_BQ, ensemble_indices_k_size)

        # k_size_max = 0

        # NOTE per_query_token_cnt_diclist is just for analysis
        if os.environ.get('ENSEMBLE_AGREE_DICLIST', '0') == '1':
            per_query_token_cnt_diclist = []
        
        # ensemble_attn_mask_per_layer [40, 128, 256, 5] to [N*H * TDST//BLOCK_SIZE_Q, mask_k//BLOCK_SIZE_K * ensemble_model_n]
        ensemble_attn_mask_per_layer = ensemble_attn_mask_per_layer.view(_N_H * TDST_BQ, MASK_K_BK * MODEL_N)
        # ensemble_attn_mask_per_layer : [N*H * TDST//BLOCK_SIZE_Q, mask_k//BLOCK_SIZE_K * ensemble_model_n]
        unique_x, indices, unique_cnt = torch.unique(ensemble_attn_mask_per_layer, return_inverse=True, sorted=False, return_counts=True)
        # indices -= indices.min(dim=1, keepdims=True)[0]
        
        # cnt_x = (unique_x[None, None, :] == ensemble_attn_mask_per_layer[:, :, None]).long().sum(1)
        N, K = ensemble_attn_mask_per_layer.shape

        cnt_xs = []
        chunk_n = 64
        for icn in range(int(math.ceil(N / chunk_n))):
            cnt_xs.append(
                (unique_x[None, None, :] == ensemble_attn_mask_per_layer[icn*chunk_n:icn*chunk_n+chunk_n, :, None]).long().sum(1)
            )
        cnt_x = torch.cat(cnt_xs, dim=0)
        
        
        result = torch.full_like(ensemble_attn_mask_per_layer, 9999999)
        result = result.scatter_(1, indices.clamp(0, K-1), ensemble_attn_mask_per_layer)
        
        
        # t = result[:, :, None] == unique_x[None, None, :]
        # t = t * torch.arange(len(unique_x), device=t.device)[None, None, :]
        # t = t.sum(-1)
        N, K = result.shape
        ts = []
        chunk_n = 64
        for icn in range(int(math.ceil(N / chunk_n))):
            t = result[icn*chunk_n:icn*chunk_n+chunk_n, :, None] == unique_x[None, None, :]
            t = t * torch.arange(len(unique_x), device=t.device)[None, None, :]
            t = t.sum(-1)
            ts.append(t)
        t = torch.cat(ts, dim=0)

        # torch.Size([4096, 1280]) torch.Size([64, 1280]) torch.Size([4096, 2094])
        # print(result.shape, t.shape, cnt_x.shape)
        result_cnt = torch.where(result < 9999999, cnt_x.gather(-1, t), -9999999)

        '''
        ensemble_attn_mask_per_layer
        tensor([[1, 1, 3, 3, 3, 5],\
                [3, 3, 4, 4, 4, 4]])
        ensemble_cnt_sorted
        tensor([[       6,        4,        3,        3,        2,        1,        1,
          -9999999, -9999999, -9999999, -9999999, -9999999, -9999999, -9999999,
          -9999999, -9999999, -9999999, -9999999, -9999999, -9999999],\
         [       6,        4,        3,        3,        2,        2, -9999999,
          -9999999, -9999999, -9999999, -9999999, -9999999, -9999999, -9999999,
          -9999999, -9999999, -9999999, -9999999, -9999999, -9999999]])
        ensemble_sorted
         tensor([[   10,     6,     3,     8,     1,     5,    11, 32000, 32000, 32000,
          32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000],\
         [    5,     4,     7,     9,     6,     3, 32000, 32000, 32000, 32000,
          32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000]]))
        '''
        # ensemble_sorted : [N*H * TDST//BLOCK_SIZE_Q, mask_k//BLOCK_SIZE_K * ensemble_model_n]
        # ensemble_cnt_sorted : [N*H * TDST//BLOCK_SIZE_Q, mask_k//BLOCK_SIZE_K * ensemble_model_n]
        ensemble_cnt_sorted, indices = torch.sort(result_cnt, dim=-1, descending=True)
        ensemble_sorted = result.gather(-1, indices)
        mask = ensemble_cnt_sorted >= ensemble_method_final_inter_thresh

        ensemble_filtered = torch.where(mask, ensemble_sorted, torch.tensor(9999999, device=mask.device))
        ensemble_cnt_filtered = torch.where(mask, ensemble_cnt_sorted, torch.tensor(9999999, device=mask.device))
        
        ## mask_i : where to discard leftovers 
        filtered_mask = ensemble_filtered == 9999999
        # Determine which columns have all rows as -1
        columns_with_all_negative_one = torch.all(filtered_mask, dim=0)

        # Get the first index where all rows have -1
        nonzero_indices = torch.nonzero(columns_with_all_negative_one, as_tuple=True)

        # If there are any such columns, find the first one
        if len(nonzero_indices[0]) > 0:
            mask_k_i = nonzero_indices[0][0].item()
            k_final = min(mask_k_i, ensemble_indices_k_size)
        else:
            mask_k_i = -1  # If no such index is found
            k_final = ensemble_indices_k_size

        ensemble_filtered = ensemble_filtered[:, :k_final] # TODO is this meaningful??
        ensemble_cnt_filtered = ensemble_cnt_filtered[:, :k_final]
        ensemble_filtered = ensemble_filtered.view(_N_H, TDST_BQ, -1)

        k_mask = ensemble_filtered < 9999999
        ks = k_mask.sum(dim=-1).view(_N_H, TDST_BQ)
        sparsity_per_layer = torch.sum(ensemble_filtered<9999999).item()
        sparsity_ratio = (sparsity_per_layer/origin_sparsity)

        # NOTE per_query_token_cnt_diclist is just for analysis
        if os.environ.get('ENSEMBLE_AGREE_DICLIST', '0') == '1':
            os.makedirs('./cache/llama/bef_ensb', exist_ok=True)
            torch.save({
                "initial_indices": result,
                "initial_cnt" : result_cnt,
                'sorted_indices' : ensemble_sorted,
                'sorted_cnt' : ensemble_cnt_sorted,
                'randomness' : ensemble_randomness,
                'final_indices' : ensemble_filtered,
                'final_cnt' : ensemble_cnt_filtered
            }, f'./cache/llama/bef_ensb/ensbn{ensemble_model_n}_agreement_0.5.pth')
            input('>>> ')

        # for r in ensemble_attn_mask_per_layer:
        #     unique_ensemble, ensemble_cnt = torch.unique(r, return_counts=True)
            
        #     ##### FOR Analysis
        #     if os.environ.get('ENSEMBLE_AGREE_DICLIST', '0') == '1':
        #         d = dict(zip(unique_ensemble.tolist(), ensemble_cnt.tolist()))
        #         sorted_dic = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))
        #         sorted_dic.pop(9999999, None)
        #         per_query_token_cnt_diclist.append(sorted_dic)
        #     #####

        #     mask = ensemble_cnt >= ensemble_method_final_inter_thresh
        #     unique_filtered = unique_ensemble[mask]
        #     counts_filtered = ensemble_cnt[mask]

        #     sorted_indices = torch.argsort(counts_filtered, descending=True)
        #     unique_filtered = unique_filtered[sorted_indices]

        #     len_uf = len(unique_filtered)
        #     if len_uf > ensemble_indices_k_size:
        #         unique_filtered = unique_filtered[:ensemble_indices_k_size]

        #     if len_uf > k_size_max:
        #         k_size_max = len_uf

        #     ensembled_indices[:,:len_uf] = unique_filtered

        # assert k_size_max <= ensemble_indices_k_size
        # # breakpoint()
        # assert torch.all(ensembled_indices[:, k_size_max:] == 9999999)
        # ensembled_indices = ensembled_indices[:, :k_size_max] # TODO : Is undoing better for padding's perspective?
        # ensembled_indices = ensembled_indices.view(_N_H, TDST_BQ, -1)

    # k_mask = ensembled_indices != 9999999
    # ks = k_mask.sum(dim=2)
    # sparsity_per_layer = torch.sum(ensembled_indices!=9999999).item()
    # sparsity_ratio = (sparsity_per_layer/origin_sparsity)

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
    #         'q_timber': q_timber,
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
    #         'ensemble_per_attn_iter_n' : ensemble_per_attn_iter_n,
    #         'ensemble_model_n' : ensemble_model_n,
    #         'ensemble_particular_layer' : ensemble_particular_layer,
    #         'layer_id' : layer_id,

    #         'ensemble_attn_mask_per_layer': ensemble_attn_mask_per_layer,
    #         'per_query_token_cnt_diclist': per_query_token_cnt_diclist,
    #         'ensembled_indices' : ensembled_indices,
    #         'origin_sparsity' : origin_sparsity,
    #         'sparsity_per_layer' : sparsity_per_layer,
    #         'sparse_ratio' : sparsity_ratio,

    #     }, f'./cache/ensemble/llama13b_32k/method/{ensemble_model_setting}_{ensemble_method}_{ensemble_method_final}/l_{layer_id}_m_{ensemble_model_n}_pl_{ensemble_per_layer_n}_pat{ensemble_per_attn_iter_n}_ln{ensemble_particular_layer}.pth')
    #     print(">>> STORED.")
        # input('stored. press enter to continue >>> ')
    # breakpoint()
    ##########
    # breakpoint()
    return ensemble_filtered, ks, origin_sparsity, sparsity_per_layer, sparsity_ratio