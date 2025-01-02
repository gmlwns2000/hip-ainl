import math
import os
import time
import cuda
import cuda.cudart
import torch
from torch import Tensor
from typing import Optional, Tuple, Union
import tqdm
import triton
import triton.language as tl

from hip.models.hip_attention.offload_runner.tensor_from_pointer import (
    tensor_from_pointer
)
from hip.models.hip_attention.gen3.attention_metadata import (
    HiPAttentionOutputMetadata,
    HiPAttentionCacheAccessStatistics,
)

MAX_INT: tl.constexpr = 2_147_483_647

def sizeof(dtype: Union[Tensor, torch.dtype]) -> int:
    if isinstance(dtype, Tensor):
        return dtype.numel() * sizeof(dtype.dtype)
    
    if dtype in [
        torch.uint8, 
        torch.int8, 
        torch.float8_e4m3fn, 
        torch.float8_e4m3fnuz, 
        torch.float8_e5m2, 
        torch.float8_e5m2fnuz
    ]:
        return 1
    elif dtype in [
        torch.uint16,
        torch.int16,
        torch.float16,
        torch.bfloat16,
    ]:
        return 2
    elif dtype in [
        torch.uint32,
        torch.int32,
        torch.float32,
    ]:
        return 4
    elif dtype in [
        torch.uint64,
        torch.int64,
        torch.float64,
    ]:
        return 8

def format_size_bytes(tensor: Union[Tensor, Union[float, int]]) -> str:
    if isinstance(tensor, Tensor):
        byte_size = sizeof(tensor)
    elif isinstance(tensor, (int, float)):
        byte_size = tensor
    else:
        raise Exception()

    if byte_size < 1024:
        return f'{byte_size} B'
    elif byte_size < (1024 ** 2):
        return f'{byte_size / 1024:.2f} KB'
    elif byte_size < (1024 ** 3):
        return f'{byte_size / (1024 ** 2):.2f} MB'
    else:
        return f'{byte_size / (1024 ** 3):.2f} GB'

def debug_print(*args):
    print(f'[HiPOffloadKVPoolMHA] {" ".join(map(lambda x: str(x), args))}')

###############################################################################
#                               Data Structure
###############################################################################

def uvm_note_cpu(tensor: Tensor, prefetch: bool = False):
    cuda.cudart.cudaMemAdvise(tensor.data_ptr(), tensor.numel() * tensor.element_size(), cuda.cudart.cudaMemoryAdvise.cudaMemAdviseSetPreferredLocation, -1)
    cuda.cudart.cudaMemAdvise(tensor.data_ptr(), tensor.numel() * tensor.element_size(), cuda.cudart.cudaMemoryAdvise.cudaMemAdviseSetAccessedBy, tensor.device.index)
    if prefetch:
        cuda.cudart.cudaMemPrefetchAsync(tensor.data_ptr(), tensor.numel() * tensor.element_size(), -1, 0)

class UVMCache:
    bank_cpu: Tensor
    bank_gpu: Tensor
    metadata: Tensor
    
    def __init__(
        self, 
        max_token_size: int,
        head_num: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.max_token_size = max_token_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        if self.device.index is None:
            self.device = torch.get_default_device()
        
        self.bank_cpu, self.bank_gpu = self.alloc_uvm(
            [max_token_size, head_num, head_dim],
            dtype=self.dtype
        )
        
        # {
        #     Token Generation: uint32    # Increase one on every overwrite
        # }
        self.metadata = torch.full(
            [max_token_size, 1], 
            dtype=torch.int32, 
            device=device,
            fill_value=MAX_INT
        )
        
        self.allocated_cpu_bytes = sizeof(self.bank_cpu)
        self.allocated_gpu_bytes = sizeof(self.metadata)
        
        # debug_print(f'UVMCache: bank={format_size_bytes(self.bank_cpu)}, metadata={format_size_bytes(self.metadata)}')
    
    def alloc_uvm(self, shape, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        device = self.device
        if isinstance(device, str):
            device = torch.device(device)
            
        elem_size = sizeof(dtype)
        numel = math.prod(shape)
        align = 4096
        byte_size = elem_size * numel
        byte_size = byte_size + byte_size % align
        
        _result_code, pointer = cuda.cudart.cudaMallocManaged(
            byte_size, 
            cuda.cudart.cudaMemAttachGlobal
        )
        
        t_gpu = tensor_from_pointer(pointer, shape, dtype, device.index)
        t_cpu = tensor_from_pointer(pointer, shape, dtype, -1)
        
        uvm_note_cpu(t_gpu)
        t_cpu.fill_(0)
        
        return t_cpu, t_gpu
    
    def gather_cpu(self, table: Tensor, pin_memory = False) -> Tensor:
        assert table.ndim == 1
        assert table.device == self.bank_cpu.device
        
        t = torch.zeros(
            (table.shape[0], self.bank_cpu.shape[1], self.bank_cpu.shape[2]), 
            dtype=self.bank_cpu.dtype, 
            device='cpu'
        )
        
        view_dtype = torch.uint16
        if self.bank_cpu.dtype in [torch.float32]:
            view_dtype = torch.uint32
        elif self.bank_cpu.dtype in [torch.float16, torch.bfloat16]:
            view_dtype = torch.uint16
        elif self.bank_cpu.dtype in [torch.uint8, torch.float8_e5m2]:
            view_dtype = torch.uint8
        else:
            raise Exception()

        index_copy(
            self.bank_cpu.view(dtype=view_dtype).numpy(), 
            t.view(dtype=view_dtype).numpy(), 
            table.numpy()
        )
        
        if pin_memory:
            t = t.pin_memory()
        
        return t

import numba
import numpy as np

@numba.njit(parallel=True, fastmath=True)
def index_copy(src: np.ndarray, out: np.ndarray, table: np.ndarray):
    for i in numba.prange(table.shape[0]):
        out[i] = src[table[i]]

def pad_to_cacheline(nelem: int, dtype: torch.dtype):
    byte_size = 4
    if dtype in [torch.int32, torch.uint32, torch.float32]:
        byte_size = 4
    elif dtype in [torch.int64, torch.uint64, torch.float64]:
        byte_size = 8
    elif dtype in [torch.int16, torch.uint16, torch.bfloat16, torch.float16]:
        byte_size = 2
    else:
        raise Exception()
    
    assert nelem > 0

    cacheline_size = 128
    step = cacheline_size // byte_size
    return nelem if (nelem % step) == 0 else (
        nelem + step - (nelem % step)
    )

class GPUCache:
    global_metadata: Tensor
    bank: Tensor
    metadata: Tensor
    table: Tensor

    def __init__(
        self, 
        k_uvm: UVMCache, 
        v_uvm: Optional[UVMCache],
        max_cache_token_size: int,
    ):
        self.k_uvm = k_uvm
        self.v_uvm = v_uvm
        self.head_num = self.k_uvm.head_num
        self.head_dim = self.k_uvm.head_dim
        self.dtype = self.k_uvm.dtype
        self.device = self.k_uvm.device
        self.kv_packed = self.v_uvm is not None
        if self.kv_packed:
            assert self.head_num == self.v_uvm.head_num
            self.head_dim += self.v_uvm.head_dim
        self.max_cache_token_size = max_cache_token_size
        self.max_uvm_token_size = self.k_uvm.max_token_size
        if self.kv_packed:
            assert self.max_uvm_token_size == self.v_uvm.max_token_size
        
        """
        [
            CachelinePadded { current_tick: int32 }
        ]
        """
        self.global_metadata = torch.zeros(
            (1, pad_to_cacheline(1, torch.int32)), 
            dtype=torch.int32, 
            device=self.device
        )
        
        self.bank = torch.zeros(
            (self.max_cache_token_size, self.head_dim), 
            dtype=self.dtype, 
            device=self.device
        )
        
        """
        CachelinePadded {
            [0] Back reference to table: int64,        # initial handshake, store token index of UVM bank
            [1] Reference to UVM Cache: int64,         # MAX_TOKEN, for token generation check
            [2] Token Generation of UVM Cache: int64,  # To check the version of cached token
            [3] Last accessed tick: int64,
        }
        """
        self.metadata = torch.full(
            (self.max_cache_token_size, pad_to_cacheline(4, torch.int64)),
            dtype=torch.int64,
            device=self.device,
            fill_value=MAX_INT,
        )
        self.metadata[:, 3].fill_(0)
        
        # NOTE: this table is way too large to pad... sorry
        self.table = torch.full(
            (self.head_num, self.max_uvm_token_size, 1),
            dtype=torch.int32,
            device=self.device,
            fill_value=MAX_INT,
        )
        
        self.allocated_gpu_bytes = (
            sizeof(self.global_metadata) +
            sizeof(self.bank) + 
            sizeof(self.metadata) + 
            sizeof(self.table)
        )

        self.flag = False
        self.step = 0

    def handle_cache_miss(
        self,
        metadata: HiPAttentionOutputMetadata,
        stats: HiPAttentionCacheAccessStatistics
    ):
        if self.flag: return
        
        # NOTE: this function should be capturable.
        # NOTE: this function will called only when mask is updated

        uvm_page_count = self.k_uvm.bank_cpu.shape[0]
        gpu_page_count = self.bank.shape[0]

        assert stats.cache_miss_counter.shape[1:] == (self.head_num, uvm_page_count), \
            f'{stats.cache_miss_counter.shape[1:]} == [{self.head_num}, {uvm_page_count}]'
    
        # update LRU recency
        # increase LRU step
        self.global_metadata[0, 0].add_(1)

        accessed = stats.access_counter.sum(0)

        assert accessed.ndim == 2
        assert accessed.shape == (self.head_num, uvm_page_count)
        assert self.k_uvm.metadata.shape == (uvm_page_count, 1)
        assert self.global_metadata.shape == (1, pad_to_cacheline(1, self.global_metadata.dtype))
        assert self.metadata.shape == (self.bank.shape[0], pad_to_cacheline(4, self.metadata.dtype))
        assert self.table.shape == (self.head_num, uvm_page_count, 1)

        BLOCK_SIZE = 128
        grid = (self.head_num * triton.cdiv(uvm_page_count, BLOCK_SIZE), )
        update_recency[grid](
            accessed, *accessed.stride(),

            self.k_uvm.metadata, *self.k_uvm.metadata.stride(),

            self.global_metadata, *self.global_metadata.stride(),
            self.metadata, *self.metadata.stride(),
            self.table, *self.table.stride(),

            uvm_page_count,

            BLOCK_SIZE,

            num_warps=4,
        )
        self.step += 1

        # perform LRU
        assert gpu_page_count <= (uvm_page_count * self.head_num), f'{gpu_page_count} <= {(uvm_page_count * self.head_num)}'

        cache_miss = ((stats.cache_miss_counter > 0) * stats.access_counter).sum(0).view(-1)
        put_mask = cache_miss > 0
        put_priority_list = cache_miss.argsort(-1, descending=True)
        put_priority_list = put_priority_list[:gpu_page_count]
        put_mask = put_mask[put_priority_list]

        slot_recency = self.metadata[:, 3]
        evict_priority_list = slot_recency.argsort(-1, descending=False)
        
        self.write_cache(
            put_list=put_priority_list,
            put_mask=put_mask,
            evict_list=evict_priority_list,
        )

        # self.flag = True

        # self._verify_cache(put_mask)
    
    def _verify_cache(self, put_mask):
        table = self.table.cpu()
        metadata = self.metadata.cpu()
        bank = self.bank.cpu()
        uvm_metadata = self.k_uvm.metadata.cpu()
        uvm_k_bank = self.k_uvm.bank_cpu
        uvm_v_bank = self.v_uvm.bank_cpu if self.kv_packed else None

        total_cache_hit = 0
        for idx_head in range(table.shape[0]):
            for idx_page in tqdm.tqdm(range(table.shape[1]), dynamic_ncols=True, leave=False):
                target_slot = table[idx_head, idx_page].item()
                if target_slot != MAX_INT:
                    back_ref, ref_to_uvm, token_gen, last_tick = metadata[target_slot, :4]
                    if (back_ref == idx_page) and (ref_to_uvm != MAX_INT) and (uvm_metadata[ref_to_uvm, 0] == token_gen):
                        gpu_value = bank[target_slot]
                        if not self.kv_packed:
                            cpu_value = uvm_k_bank[idx_page, idx_head]
                        else:
                            cpu_value = torch.cat([
                                uvm_k_bank[idx_page, idx_head],
                                uvm_v_bank[idx_page, idx_head],
                            ], dim=0)
                        mse = ((gpu_value - cpu_value) ** 2).mean().item()
                        assert mse < 1e-4, mse
                        assert last_tick > 0, last_tick
                        total_cache_hit += 1
        print('verified', total_cache_hit, 'lastly put', put_mask.sum().item())

    def write_cache(
        self,
        put_list: Tensor,
        put_mask: Tensor,
        evict_list: Tensor,
    ):
        assert put_list.shape == put_mask.shape
        assert evict_list.shape == put_list.shape

        BLOCK_SIZE = 128

        qsize = put_list.shape[0]

        grid = (triton.cdiv(qsize, BLOCK_SIZE),)
        write_cache[grid](
            put_list, *put_list.stride(),
            put_mask, *put_mask.stride(),
            evict_list, *evict_list.stride(),

            self.bank, *self.bank.stride(),
            self.metadata, *self.metadata.stride(),
            self.table, *self.table.stride(),

            self.k_uvm.metadata, *self.k_uvm.metadata.stride(),
            self.k_uvm.bank_gpu, *self.k_uvm.bank_gpu.stride(),
            self.v_uvm.bank_gpu if self.kv_packed else None, 
            *(self.v_uvm.bank_gpu.stride() if self.kv_packed else (0, 0, 0)),

            self.global_metadata, *self.global_metadata.stride(),

            qsize,
            self.k_uvm.bank_gpu.shape[0],

            self.kv_packed,
            BLOCK_SIZE,
            self.k_uvm.bank_gpu.shape[-1]
        )

class HiPOffloadCache:
    def __init__(
        self,
        max_token_size: int,
        max_mask_cache_token_size: int,
        max_sa_cache_token_size: int,
        head_num: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.k_uvm = UVMCache(
            max_token_size=max_token_size,
            head_num=head_num,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
        
        self.v_uvm = UVMCache(
            max_token_size=max_token_size,
            head_num=head_num,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
        
        self.mask_k_cache = GPUCache(
            k_uvm=self.k_uvm,
            v_uvm=None,
            max_cache_token_size=max_mask_cache_token_size,
        )
        
        self.sa_kv_cache = GPUCache(
            k_uvm=self.k_uvm,
            v_uvm=self.v_uvm,
            max_cache_token_size=max_sa_cache_token_size,
        )
    
    def get_page_count(self):
        assert self.k_uvm.bank_cpu.shape == self.v_uvm.bank_cpu.shape
        return self.k_uvm.bank_cpu.shape[0]
    
    def prefetch_prefix_kv_buffer(
        self,
        table: Tensor,
        device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        # t = time.time()
        table = table.to('cpu', non_blocking=False)
        # e1 = time.time()
        k = self.k_uvm.gather_cpu(table, pin_memory=True)
        v = self.v_uvm.gather_cpu(table, pin_memory=True)
        # e2 = time.time()
        k = k.to(device, non_blocking=False).unsqueeze(0)
        v = v.to(device, non_blocking=False).unsqueeze(0)
        # e3 = time.time()
        # print(e1-t, e2-e1, e3-e2)
        return k, v
    
    def set_kv_buffer(
        self,
        table: torch.Tensor,
        table_gpu: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        cache_device = cache_k.device
        assert table.device == cache_device
        assert cache_v.device == cache_device
        
        if cache_device == torch.device('cpu'):
            self.k_uvm.bank_cpu[table] = cache_k
            self.v_uvm.bank_cpu[table] = cache_v
        else:
            assert cache_device == self.k_uvm.device
            self.k_uvm.bank_gpu[table] = cache_k
            self.v_uvm.bank_gpu[table] = cache_v
        
        self.k_uvm.metadata.index_put_(
            indices=(table, ), 
            values=torch.index_select(self.k_uvm.metadata, index=table_gpu, dim=0) + 1
        )
        self.v_uvm.metadata.index_put_(
            indices=(table, ), 
            values=torch.index_select(self.v_uvm.metadata, index=table_gpu, dim=0) + 1
        )
    
    def handle_cache_miss(self, metadata: HiPAttentionOutputMetadata):
        if metadata.mask_cache_statistics is not None:
            self.mask_k_cache.handle_cache_miss(
                metadata=metadata,
                stats=metadata.mask_cache_statistics
            )
            self.sa_kv_cache.handle_cache_miss(
                metadata=metadata,
                stats=metadata.sa_cache_statistics
            )

###############################################################################
#                               Kernel Function
###############################################################################

@triton.jit
def load_tokens(
    K, 
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head,
    stride_k_hid,
    
    # paged attention args template
    USING_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_CACHE, 
    stride_k_cache_page, 
    stride_k_cache_offset, 
    stride_k_cache_kv_head, 
    stride_k_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_b,
    
    USING_OFFLOAD_CACHE: tl.constexpr,
    OFFLOAD_CACHE_KV_PACKED: tl.constexpr,
    OFFLOAD_CACHE_LOAD_VALUE: tl.constexpr,
    OFFLOAD_CACHE_UVM_METADATA,
    stride_offload_cache_uvm_metadata_token,
    stride_offload_cache_uvm_metadata_k,
    OFFLOAD_CACHE_GPU_BANK,
    stride_offload_cache_gpu_bank_token,
    stride_offload_cache_gpu_bank_hid,
    OFFLOAD_CACHE_GPU_METADATA,
    stride_offload_cache_gpu_metadata_token,
    stride_offload_cache_gpu_metadata_k,
    OFFLOAD_CACHE_GPU_TABLE,
    stride_offload_cache_gpu_table_head_kv,
    stride_offload_cache_gpu_table_token,
    strdie_offload_cache_gpu_table_k,
    
    ACCESS_COUNTER,
    stride_access_counter_bsz,
    stride_access_counter_head_kv,
    stride_access_counter_tsrc,
    CACHE_MISS_COUNTER,
    stride_cache_miss_counter_bsz,
    stride_cache_miss_counter_head_kv,
    stride_cache_miss_counter_tsrc,
    
    idx_bsz,
    idx_tsrc,
    idx_kv_head,
    idx_hid,
    
    mask_keys,
    
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    # DEBUG: to load nothing
    # mask_keys = mask_keys & False
    
    # tl.static_print(OFFLOAD_CACHE_METHOD)
    
    if not USING_PAGES:
        tl.static_assert(not USING_OFFLOAD_CACHE)
        
        tl.atomic_add(
            ACCESS_COUNTER +\
                idx_bsz.to(tl.int64) * stride_access_counter_bsz +\
                idx_kv_head * stride_access_counter_head_kv +\
                idx_tsrc * stride_access_counter_tsrc,
            mask=mask_keys,
            val=1
        )
        
        keys = tl.load(
            K +\
                idx_bsz.to(tl.int64) * stride_k_bsz +\
                idx_tsrc.to(tl.int64) * stride_k_tsrc +\
                idx_kv_head.to(tl.int64) * stride_k_head +\
                idx_hid.to(tl.int64) * stride_k_hid,
            mask = mask_keys,
            other = 0.0,
            # cache_modifier='.cs', # TODO: uncomment this
        )
    else:
        seq_len = tl.load(
            CACHE_SEQ_LENS +\
                idx_bsz.to(tl.int64) * stride_cache_seq_lens_b,
        )
        mask_tsrc = idx_tsrc < seq_len
        ptrs = BLOCK_TABLE +\
            idx_bsz.to(tl.int64) * stride_block_table_bsz + \
            (idx_tsrc // PAGE_SIZE).to(tl.int64) * stride_block_table_page
        idx_page = tl.load(
            ptrs,
            mask=mask_tsrc,
            other=0,
        ).to(tl.int64)
        offset_page = idx_tsrc % PAGE_SIZE
        
        tl.atomic_add(
            ACCESS_COUNTER +\
                idx_bsz.to(tl.int64) * stride_access_counter_bsz +\
                idx_kv_head * stride_access_counter_head_kv +\
                idx_page * stride_access_counter_tsrc,
            mask=mask_keys,
            val=1
        )
        
        if USING_OFFLOAD_CACHE:
            tl.static_assert(PAGE_SIZE == 1)
            original_mask_keys = mask_keys
            
            idx_slots = tl.load(
                OFFLOAD_CACHE_GPU_TABLE +\
                    idx_page * stride_offload_cache_gpu_table_token +\
                    idx_kv_head * stride_offload_cache_gpu_table_head_kv +\
                    0 * strdie_offload_cache_gpu_table_k,
                mask=mask_keys,
                other=MAX_INT,
            )
            idx_slot_has_reference_to_bank = idx_slots != MAX_INT
            
            slot_metadata_backref_to_table = tl.load(
                OFFLOAD_CACHE_GPU_METADATA +\
                    idx_slots * stride_offload_cache_gpu_metadata_token +\
                    0 * stride_offload_cache_gpu_metadata_k,
                mask=idx_slot_has_reference_to_bank,
                other=MAX_INT,
            )
            idx_slot_is_valid_link = (
                slot_metadata_backref_to_table == idx_page
            ) & idx_slot_has_reference_to_bank
            
            slot_metadata_ref_to_uvm = tl.load(
                OFFLOAD_CACHE_GPU_METADATA +\
                    idx_slots * stride_offload_cache_gpu_metadata_token +\
                    1 * stride_offload_cache_gpu_metadata_k,
                mask=idx_slot_is_valid_link,
                other=MAX_INT,
            )
            slot_metadata_token_gen = tl.load(
                OFFLOAD_CACHE_GPU_METADATA +\
                    idx_slots * stride_offload_cache_gpu_metadata_token +\
                    2 * stride_offload_cache_gpu_metadata_k,
                mask=idx_slot_is_valid_link,
                other=MAX_INT,
            )
            idx_slot_is_valid_link = (
                slot_metadata_ref_to_uvm != MAX_INT
            ) & idx_slot_is_valid_link
            
            uvm_metadata_token_gen = tl.load(
                OFFLOAD_CACHE_UVM_METADATA +\
                    slot_metadata_ref_to_uvm * stride_offload_cache_uvm_metadata_token +\
                    0 * stride_offload_cache_uvm_metadata_k,
                mask=idx_slot_is_valid_link,
                other=MAX_INT
            )
            
            mask_slot_cache_hit = (
                uvm_metadata_token_gen != MAX_INT
            ) & (
                uvm_metadata_token_gen == slot_metadata_token_gen
            ) & idx_slot_is_valid_link
            
            idx_hid_cached = idx_hid
            if OFFLOAD_CACHE_LOAD_VALUE:
                idx_hid_cached += BLOCK_HID
            keys_cached = tl.load(
                OFFLOAD_CACHE_GPU_BANK +\
                    idx_slots * stride_offload_cache_gpu_bank_token +\
                    idx_hid_cached * stride_offload_cache_gpu_bank_hid,
                mask=mask_slot_cache_hit,
                other=0.0,
            )
            if keys_cached.dtype == tl.uint8:
                keys_cached = keys_cached.to(tl.float8e5, bitcast=True).to(tl.bfloat16)
            if keys_cached.dtype == tl.float8e5:
                keys_cached = keys_cached.to(tl.bfloat16)
            
            mask_keys = mask_keys & (~mask_slot_cache_hit)
        
        keys = tl.load(
            K_CACHE +\
                idx_page.to(tl.int64) * stride_k_cache_page +\
                offset_page.to(tl.int64) * stride_k_cache_offset +\
                idx_kv_head.to(tl.int64) * stride_k_cache_kv_head +\
                idx_hid.to(tl.int64) * stride_k_cache_hid,
            mask=mask_keys,
            other=0.0,
        )
        if keys.dtype == tl.uint8:
            keys = keys.to(tl.float8e5, bitcast=True).to(tl.bfloat16)
        if keys.dtype == tl.float8e5:
            keys = keys.to(tl.bfloat16)
        
        if USING_OFFLOAD_CACHE:
            keys = tl.where(
                mask_slot_cache_hit,
                keys_cached,
                keys,
            )
            
            tl.atomic_add(
                CACHE_MISS_COUNTER +\
                    idx_bsz.to(tl.int64) * stride_cache_miss_counter_bsz +\
                    idx_kv_head * stride_cache_miss_counter_head_kv +\
                    idx_page * stride_cache_miss_counter_tsrc,
                mask=mask_keys,
                val=1,
            )
    
    if keys.dtype == tl.uint8:
        keys = keys.to(tl.float8e5, bitcast=True).to(tl.float16)
    if keys.dtype == tl.float8e5:
        keys = keys.to(tl.float16)
    
    return keys

def update_recency_pytorch(
    accessed_ptr: Tensor,
    uvm_metadata: Tensor,
    global_metadata: Tensor,
    metadata: Tensor,
    table: Tensor,
    head_num: int,
    uvm_page_count: int,
):
    for idx_head_kv in range(head_num):
        for idx_token in tqdm.tqdm(range(uvm_page_count), dynamic_ncols=True, leave=False):
            current_tick = global_metadata[0, 0]
            
            accessed = accessed_ptr[idx_head_kv, idx_token] > 0
            cache_hit = True & accessed
            if not cache_hit: continue

            idx_table = table[idx_head_kv, idx_token]
            cache_hit = (idx_table != MAX_INT) & cache_hit
            if not cache_hit: continue
            
            back_ref = metadata[idx_table, 0]
            cache_hit = (back_ref == idx_token) & cache_hit
            if not cache_hit: continue

            ref_to_uvm = metadata[idx_table, 1]
            cache_hit = (ref_to_uvm != MAX_INT) & cache_hit
            if not cache_hit: continue

            uvm_token_gen = uvm_metadata[ref_to_uvm, 0]
            cache_token_gen = metadata[idx_table, 2]
            cache_hit = (
                uvm_token_gen != MAX_INT
            ) & (
                uvm_token_gen == cache_token_gen
            ) & cache_hit
            if not cache_hit: continue

            metadata[idx_table, 3] = current_tick.to(metadata.dtype)

@triton.jit
def update_recency(
    ACCESSED,
    stride_accessed_head_kv, stride_accessed_token,

    UVM_METADATA,
    stride_uvm_metadata_token, stride_uvm_metadata_k,

    GLOBAL_METADTA,
    stride_global_metadata_k, stride_global_metadata_pad,
    METADATA,
    stride_metadata_slot, stride_metadata_k,
    TABLE,
    stride_table_head_kv, stride_table_token, stride_table_k,

    page_count: int,

    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    idx_block = pid % tl.cdiv(page_count, BLOCK_SIZE)
    idx_head_kv = pid // tl.cdiv(page_count, BLOCK_SIZE)

    idx_token = idx_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_token = idx_token < page_count

    current_tick = tl.load(
        GLOBAL_METADTA +\
            0 * stride_global_metadata_k+\
            0 * stride_global_metadata_pad
    )

    #TODO: merge with load tokens, verify cache
    accessed = tl.load(
        ACCESSED +\
            idx_head_kv * stride_accessed_head_kv +\
            idx_token * stride_accessed_token,
        mask=mask_token,
        other=0,
    ) > 0
    cache_hit = mask_token & accessed

    table = tl.load(
        TABLE +\
            idx_head_kv * stride_table_head_kv +\
            idx_token * stride_table_token +\
            0 * stride_table_k,
        mask=cache_hit,
        other=MAX_INT,
    ).to(tl.int64)
    cache_hit = (table != MAX_INT) & cache_hit
    
    back_ref = tl.load(
        METADATA +\
            table * stride_metadata_slot +\
            0 * stride_metadata_k,
        mask=cache_hit,
        other=MAX_INT
    )
    cache_hit = (back_ref == idx_token) & cache_hit

    ref_to_uvm = tl.load(
        METADATA +\
            table * stride_metadata_slot +\
            1 * stride_metadata_k,
        mask=cache_hit,
        other=MAX_INT,
    ).to(tl.int64)
    cache_hit = (ref_to_uvm != MAX_INT) & cache_hit

    uvm_token_gen = tl.load(
        UVM_METADATA +\
            ref_to_uvm * stride_uvm_metadata_token +\
            0 * stride_uvm_metadata_k,
        mask=cache_hit,
        other=MAX_INT
    )
    cache_token_gen = tl.load(
        METADATA +\
            table * stride_metadata_slot +\
            2 * stride_metadata_k,
        mask=cache_hit,
        other=MAX_INT,
    )
    cache_hit = (
        uvm_token_gen != MAX_INT
    ) & (
        uvm_token_gen == cache_token_gen
    ) & cache_hit

    tl.store(
        METADATA +\
            table * stride_metadata_slot +\
            3 * stride_metadata_k,
        mask=cache_hit,
        value=current_tick,
    )

@triton.jit
def write_cache(
    PUT, stride_put_t,
    MASK, stride_mask_t,
    EVICT, stride_evict_t,

    BANK, 
    stride_bank_t, 
    stride_bank_hid,
    METADATA, 
    stride_metadata_t, 
    stride_metadata_k,
    TABLE, 
    stride_table_head_kv, 
    stride_table_t, 
    stride_table_k,

    UVM_METADATA,
    stride_uvm_metadata_t, 
    stride_uvm_metadata_k,
    UVM_K_BANK,
    stride_uvm_k_bank_t, 
    stride_uvm_k_bank_head_kv, 
    stride_uvm_k_bank_hid,
    UVM_V_BANK,
    stride_uvm_v_bank_t, 
    stride_uvm_v_bank_head_kv, 
    stride_uvm_v_bank_hid,

    GLOBAL_METADATA,
    stride_global_metadata_t,
    stride_global_metadata_k,

    qsize: int,
    page_count: int,
    
    KV_PACKED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    pid = tl.program_id(0)
    idx_queue = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_queue = idx_queue < qsize

    put_list = tl.load(
        PUT + idx_queue * stride_put_t,
        mask=mask_queue,
    )
    idx_page = put_list % page_count
    idx_head_kv = put_list // page_count

    mask_put = (tl.load(
        MASK + idx_queue * stride_mask_t,
        mask=mask_queue,
        other=0
    ) != 0)
    idx_evict = tl.load(
        EVICT + idx_queue * stride_evict_t,
        mask=mask_put
    )

    # setup metadata
    tl.store(
        METADATA +\
            idx_evict * stride_metadata_t +\
            0 * stride_metadata_k,
        mask=mask_put,
        value=idx_page
    )
    tl.store(
        METADATA +\
            idx_evict * stride_metadata_t +\
            1 * stride_metadata_k,
        mask=mask_put,
        value=idx_page,
    )
    token_gen = tl.load(
        UVM_METADATA +\
            idx_page * stride_uvm_metadata_t +\
            0 * stride_uvm_metadata_k,
        mask=mask_put,
    )
    tl.store(
        METADATA +\
            idx_evict * stride_metadata_t +\
            2 * stride_metadata_k,
        mask=mask_put,
        value=token_gen,
    )
    current_tick = tl.load(
        GLOBAL_METADATA +\
            0 * stride_global_metadata_t +\
            0 * stride_global_metadata_k,
    )
    tl.store(
        METADATA +\
            idx_evict * stride_metadata_t +\
            3 * stride_metadata_k,
        mask=mask_put,
        value=current_tick,
    )

    # setup table
    tl.store(
        TABLE +\
            idx_page * stride_table_t +\
            idx_head_kv * stride_table_head_kv +\
            0 * stride_table_k,
        mask=mask_put,
        value=idx_evict,
    )

    # copy values
    idx_hid = tl.arange(0, BLOCK_HID)

    keys = tl.load(
        UVM_K_BANK +\
            idx_page[:, None] * stride_uvm_k_bank_t +\
            idx_head_kv[:, None] * stride_uvm_k_bank_head_kv +\
            idx_hid[None, :] * stride_uvm_k_bank_hid,
        mask=mask_put[:, None],
    )
    tl.store(
        BANK +\
            idx_evict[:, None] * stride_bank_t +\
            idx_hid[None, :] * stride_bank_hid,
        mask=mask_put[:, None],
        value=keys,
    )

    if KV_PACKED:
        values = tl.load(
            UVM_V_BANK +\
                idx_page[:, None] * stride_uvm_v_bank_t +\
                idx_head_kv[:, None] * stride_uvm_v_bank_head_kv +\
                idx_hid[None, :] * stride_uvm_v_bank_hid,
            mask=mask_put[:, None],
        )
        tl.store(
            BANK +\
                idx_evict[:, None] * stride_bank_t +\
                (idx_hid + BLOCK_HID)[None, :] * stride_bank_hid,
            mask=mask_put[:, None],
            value=values,
        )