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
BQ = 32
BK = 2
MASK_K = 512

@numba.njit
def convert_to_dense(indices, ks, TDST, TSRC, BQ, BK, MASK_K):
    mask = np.zeros((TDST, TSRC))
    for i in range(TDST // BQ):
        kk = ks[0, i]
        for j in range(MASK_K // BK):
            if j < kk:
                t = indices[0, i, j]
                mask[i*BQ:i*BQ+BQ, t:t+BK] = 1
    return mask

import matplotlib.cm as cm

def render_plot(cache_path, name, iteration):
    data = torch.load(cache_path, map_location='cpu')
    data = data['metadata']

    indices = data.indices.numpy()
    ks = data.ks.numpy()
    
    ws = np.full((TDST, ), MASK_K) * (2**max(0, iteration-1))
    tsrcs = np.arange(TSRC - TDST, TSRC)
    tsrcs = tsrcs - (tsrcs % BQ) + BQ
    ws = np.minimum(tsrcs, ws)
    
    # scales = tsrcs / ws
    
    mask = convert_to_dense(indices, ks, TDST, TSRC, BQ, BK, MASK_K)
    
    for i in range(TDST):
        # scale = scales[i]
        row = mask[i:i+1, :]
        row_resize = cv2.resize(row, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_NEAREST) # fx=scale, 
        mask[i:i+1, :] = row_resize[:, :TSRC]
    
    root = './saves/plot_hip'
    path = os.path.join(root, f'{name}.png')
    os.makedirs(root, exist_ok=True)
    
    plt.figure(figsize=(4, 3))
    plt.imshow(mask, cmap='summer')
    plt.savefig(path, dpi=400, bbox_inches='tight')
    
    print('saved', path)

if __name__ == '__main__':
    for i in range(32):
        render_plot(f'./cache/llama/metadata_l{i}.pth', f'mask_l{i}', 0)
    # render_plot('./saves/attention1_block_gpu/checkout_mask_1.pth', 'mask_1', 1)
    # render_plot('./saves/attention1_block_gpu/checkout_mask_2.pth', 'mask_2', 2)
    # render_plot('./saves/attention1_block_gpu/checkout_mask_3.pth', 'mask_3', 3)
    # render_plot('./saves/attention1_block_gpu/checkout_mask_4.pth', 'mask_4', 4)