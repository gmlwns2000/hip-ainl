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

TDST = 8192
TSRC = 8192
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

import matplotlib.cm as cm

from hip.models.hip_attention.attention1_block_gpu import to_dense

def render_plot(cache_path, layer, name):
    data = torch.load(cache_path, map_location='cpu')
    data = data['metadata']

    indices = data.indices.numpy()
    ks = data.ks.numpy()

    mask = to_dense(indices, ks, None, ks.shape[0], TDST, TSRC, BQ, BK)
    
    mask_image = mask * 255
    
    # for i in range(TDST):
    #     # scale = scales[i]
    #     row = mask[i:i+1, :]
    #     row_resize = cv2.resize(row, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_NEAREST) # fx=scale, 
    #     mask[i:i+1, :] = row_resize[:, :TSRC]
    
    root = f'./saves__/plot_l{layer}'
    os.makedirs(root, exist_ok=True)
    
    for i in range(mask.shape[0]):
        img_path = os.path.join(root, f'{name}_t8192_{i}.png')
        
        cv2.imwrite(img_path, mask_image[i])
    
    tensor_path = os.path.join(root, f'{name}_t8192.pth')
    torch.save({
        # 'mask':mask,
        'sum':mask.sum()
    }, tensor_path)
    
    print('saved', img_path)

if __name__ == '__main__':
    for s in ['multi', 'default']: # 'multi', 
        for i in range(3):
            render_plot(f'./cache/llama_{s}/metadata_l{i}_t8192.pth', i, f'hip_{s}')
    # render_plot('./saves/attention1_block_gpu/checkout_mask_1.pth', 'mask_1', 1)
    # render_plot('./saves/attention1_block_gpu/checkout_mask_2.pth', 'mask_2', 2)
    # render_plot('./saves/attention1_block_gpu/checkout_mask_3.pth', 'mask_3', 3)
    # render_plot('./saves/attention1_block_gpu/checkout_mask_4.pth', 'mask_4', 4)