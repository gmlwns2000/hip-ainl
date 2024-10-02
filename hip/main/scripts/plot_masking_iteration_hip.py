import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numba
from hip.utils import setup_seaborn
setup_seaborn(axis_below=True)

"""
--method hip 
--k 512 
--block_size_q 64 
--block_stride_q 2 
--block_size_k 2 
--block_stride_k 1 
--stride 8192 
--dense_layers 0 
--no_quantize 
--overwrite 
--model llama3.1_8b 
--multi-branch-ratio 4 
--multi-branch-layer-all 
--multi-branch-ret-ratio-select-all 
--branching_method random 
--multi-branch-true-iter_str center
"""

TDST = 16384
TSRC = 16384
BH = 32
BQ = 64
BK = 2
MASK_K = 512
MULTI_BRANCH_ON_LAYER = 4

@numba.njit
def convert_to_dense(indices, ks, TDST, TSRC, BQ, BK, MASK_K):
    # NOTE only print 0th head
    # indices : [BSZ * BH, TDST//BQ, MASK_K//BK * MULTI_BRANCH_ON_LAYER]
    # ks : [BSZ * BH, TDST//BQ] <- values till MASK_K//BK * MULTI_BRANCH_ON_LAYER
    
    mask = np.zeros((TDST, TSRC))
    for i in range(TDST // BQ):
        kk = ks[0, i]
        for j in range(MASK_K // BK * MULTI_BRANCH_ON_LAYER):
            if j < kk:
                t = indices[0, i, j]
                mask[i*BQ:i*BQ+BQ, t:t+BK] = 1
    return mask

@numba.njit
def to_dense_multi_branch(indices_mul, ks, layer, name, root):
    mask_mul = []
    
    for i in range(MULTI_BRANCH_ON_LAYER):
        mask = to_dense(indices_mul[:, :, :, i], ks, None, ks.shape[0], TDST, TSRC, BQ, BK)
        mask_mul.append(mask)

        mask_image = mask * 255
        
        for j in range(mask.shape[0]):
            # img_path = os.path.join(root, f'{name}_t{TSRC}_m{i}_h{j}.png')
            img_path = root + f'/{name}_t{TSRC}_m{i}_h{j}.png'
            
            cv2.imwrite(img_path, mask_image[j])
    
    return mask_mul

import matplotlib.cm as cm

from hip.models.hip_attention.attention1_block_gpu import to_dense

def render_plot(cache_path, layer, name):
    data = torch.load(cache_path, map_location='cpu')
    # data = data['metadata']
    # indices = data.indices.numpy()
    # ks = data.ks.numpy()
    
    # indices : [BSZ * BH, TDST//BQ, MASK_K//BK * MULTI_BRANCH_ON_LAYER]
    # ks : [BSZ * BH, TDST//BQ] <- values till MASK_K//BK * MULTI_BRANCH_ON_LAYER
    
    indices = data['indices'].numpy()
    ks = data['ks'].numpy()
    ks_count = data['ks_count']
    ks_start_end = data['ks_start_end']
    # breakpoint()
    mask = to_dense(indices, ks, None, ks.shape[0], TDST, TSRC, BQ, BK)
    mask_image = mask * 255
    
    root = f'./saves__/plot_l{layer}'
    os.makedirs(root, exist_ok=True)
    
    for i in range(mask.shape[0]):
        img_path = os.path.join(root, f'{name}_t{TSRC}_{i}.png')
        
        cv2.imwrite(img_path, mask_image[i])
    
    breakpoint()
    tensor_path = os.path.join(root, f'{name}_t{TSRC}.pth')
    torch.save({
        # 'mask':mask,
        'sum':mask.sum()
    }, tensor_path)
    
    print('saved', img_path)

    indices = data['indices'].numpy()
    ks = data['ks'].numpy()
    ks_count = data['ks_count']
    ks_start_end = data['ks_start_end']

    # breakpoint()
    # BDST = TDST//BQ
    # BLOCK_MASK_K = MASK_K//BK
    # assert BH == indices.shape[0]
    # assert BDST == indices.shape[1]
    # assert BLOCK_MASK_K * MULTI_BRANCH_ON_LAYER == indices.shape[-1]
    
    # indices_mul = indices.reshape(BH, BDST, BLOCK_MASK_K, MULTI_BRANCH_ON_LAYER)
    # root = f'./saves/plot_l{layer}'
    
    # os.makedirs(root, exist_ok=True)
    
    # # mask_mul = to_dense_multi_branch(indices_mul, ks, layer, name, root)
    # mask_mul = []
    # for i in range(MULTI_BRANCH_ON_LAYER):
    #     mask = to_dense(indices_mul[:, :, :, i], ks, None, ks.shape[0], TDST, TSRC, BQ, BK)
    #     mask_mul.append(mask)

    #     mask_image = mask * 255
        
    #     root = f'./saves/plot_l{layer}'
    #     os.makedirs(root, exist_ok=True)
        
    #     # for j in range(5): # mask.shape[0]
    #     #     img_path = os.path.join(root, f'{name}_t{TSRC}_m{i}_h{j}.png')
            
    #     #     cv2.imwrite(img_path, mask_image[j])
            
    #     #     print('saved', img_path)
    
    # tensor_path = os.path.join(root, f'{name}_t{TSRC}.pth')
    
    # # import pickle
    # # pickle.dump(mask_mul, open(tensor_path, 'w'), protocol=4)
    # torch.save(mask_mul, tensor_path, pickle_protocol=4)
    
    # print('>> saved', tensor_path)

if __name__ == '__main__':
    for s in ['multi']: # 'multi', 'default'
        for i in range(1):
            render_plot(f'./cache/llama_{s}/metadata_l{i}_t16384.pth', i, f'hip_{s}')