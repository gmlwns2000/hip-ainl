import random
# import torch
import openvino as ov
import openvino.properties as props
from openvino.runtime import op, opset15
import numpy as np
import time

import tqdm

def seed(random_seed=42):
    # torch.manual_seed(random_seed)
    # torch.cuda.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
seed()

N, TDST, TSRC, H, HKV, D = 1, 1, 131072, 8, 8, 128

queries = np.random.randn(N, TDST, H, D).astype(np.float16)
keys = np.random.randn(N, TSRC, HKV, D).astype(np.float16)
values = np.random.randn(N, TSRC, HKV, D).astype(np.float16)
pos_tdst = np.zeros((N, TDST, ), dtype=np.int64) + TSRC - 1

NSINK = 64
NSLIDING = 2048
BLOCK_SIZE_Q = min(TDST, 16)

IS_SDPA = False

def create_model() -> ov.Model:
    # input
    input_q_shape = [N, TDST, H, D]
    q_node = op.Parameter(ov.Type.f16, ov.Shape(input_q_shape))
    input_k_shape = [N, TSRC, HKV, D]
    k_node = op.Parameter(ov.Type.f16, ov.Shape(input_k_shape))
    input_v_shape = [N, TSRC, HKV, D]
    v_node = op.Parameter(ov.Type.f16, ov.Shape(input_v_shape))
    input_pos_tdst_shape = [N, TDST]
    pos_tdst_node = op.Parameter(ov.Type.i64, ov.Shape(input_pos_tdst_shape))

    # body
    out_cum = None
    for idx_layer in range(16):
        if IS_SDPA:
            out_node = opset15.scaled_dot_product_attention(
                query=opset15.transpose(opset15.add(q_node, op.Constant(np.array(idx_layer, dtype=np.float16))), [0, 2, 1, 3]),
                key=opset15.transpose(k_node, [0, 2, 1, 3]),
                value=opset15.transpose(v_node, [0, 2, 1, 3]),
            )
            out_node = opset15.transpose(out_node, [0, 2, 1, 3])
            if out_cum is None:
                out_cum = out_node
            out_cum = out_cum + out_node
        else:
            out_bsz = []
            for idx_n in range(N):
                out_tdsts = []
                for idx_tdst in range(0, TDST, BLOCK_SIZE_Q):
                    out_heads = []
                    for idx_head in range(H):
                        # BLOCK_SIZE_Q x D
                        
                        # queries = q_node[idx_n, idx_tdst:idx_tdst+BLOCK_SIZE_Q, idx_head, :]
                        queries = opset15.slice(
                            opset15.add(q_node, op.Constant(np.array(idx_layer, dtype=np.float16))), 
                            [idx_n], [idx_n+1], [1], axes=[0]
                        )
                        queries = opset15.slice(queries, [idx_tdst], [idx_tdst+BLOCK_SIZE_Q], [1], axes=[1])
                        queries = opset15.slice(queries, [idx_head], [idx_head+1], [1], axes=[2])
                        queries = opset15.reshape(queries, ov.Shape([1, BLOCK_SIZE_Q, D]), special_zero=False)

                        pos_tdst = opset15.slice(pos_tdst_node, [idx_n], [idx_n+1], [1], axes=[0])
                        pos_tdst = opset15.slice(pos_tdst, [idx_tdst], [idx_tdst+1], [1], axes=[1])
                        pos_tdst = opset15.reshape(idx_tdst, ov.Shape([1,]), special_zero=False)
                        idx_keys_sink = opset15.range(0, NSINK, 1, ov.Type.i64)
                        idx_keys_sw = opset15.add(opset15.range(0, NSLIDING, 1, ov.Type.i64), opset15.subtract(pos_tdst, NSLIDING + idx_layer))
                        idx_keys = opset15.concat([idx_keys_sink, idx_keys_sw], axis=0)
                        idx_keys = opset15.reshape(idx_keys, ov.Shape([NSINK + NSLIDING]), special_zero=False)
                        # NOTE: idx_keys acts like page table
                        
                        #keys = k_node[idx_n, idx_keys, idx_head // (H // HKV), :]
                        keys = k_node
                        keys = opset15.slice(keys, [idx_n], [idx_n+1], [1], axes=[0])
                        keys = opset15.slice(keys, [idx_head//(H//HKV)], [idx_head//(H//HKV)+1], [1], axes=[2])
                        keys = opset15.gather(keys, idx_keys, axis=1)
                        # keys = opset15.slice(keys, [0], [NSINK + NSLIDING], [1], axes=[1])
                        keys = opset15.reshape(keys, ov.Shape([1, NSINK + NSLIDING, D]), special_zero=False)

                        #values = v_node[idx_n, idx_keys, idx_head // (H // HKV), :]
                        values = v_node
                        values = opset15.slice(values, [idx_n], [idx_n+1], [1], axes=[0])
                        values = opset15.slice(values, [idx_head//(H//HKV)], [idx_head//(H//HKV)+1], [1], axes=[2])
                        values = opset15.gather(values, idx_keys, axis=1)
                        # values = opset15.slice(values, [0], [NSINK + NSLIDING], [1], axes=[1])
                        values = opset15.reshape(values, ov.Shape([1, NSINK + NSLIDING, D]), special_zero=False)

                        # fused
                        out = opset15.scaled_dot_product_attention(
                            query=queries,
                            key=keys,
                            value=values,
                        )

                        # vs. vanila
                        # out = opset15.matmul(queries, keys, False, True)
                        # out = opset15.softmax(out, axis=-1)
                        # out = opset15.matmul(out, values, False, False)

                        out = opset15.reshape(out, ov.Shape([1, BLOCK_SIZE_Q, 1, D]), special_zero=False)

                        out_heads.append(out)
                    out_heads = opset15.concat(out_heads, axis=2)
                    out_tdsts.append(out_heads)
                out_tdsts = opset15.concat(out_tdsts, axis=1)
                out_bsz.append(out_tdsts)
            out_node = opset15.concat(out_bsz, axis=0, name='out_node')
            if out_cum is None:
                out_cum = out_node
            out_cum = out_cum + out_node
    
    out_node = out_cum

    return ov.Model(
        [
            out_node
        ],
        [
            q_node, 
            k_node, 
            v_node, 
            pos_tdst_node,
        ], 
        'bsa'
    )

model = create_model()

core = ov.Core()

# available_devices = core.available_devices
# print(available_devices)

device = 'GPU'

config = {
    props.hint.performance_mode: props.hint.PerformanceMode.THROUGHPUT,
    # props.hint.performance_mode: props.hint.PerformanceMode.LATENCY,
    props.hint.inference_precision: ov.Type.f16,
}
compiled_model = core.compile_model(model, device, config=config)
print(compiled_model)

request = compiled_model.create_infer_request()

if device in  ['GPU', 'NPU']:
    context = core.get_default_context(device) # openvino._pyopenvino.RemoteContext

    remote_query_tensor = context.create_tensor(ov.Type.f16, ov.Shape(tuple(queries.shape)), {})
    remote_key_tensor = context.create_tensor(ov.Type.f16, ov.Shape(tuple(keys.shape)), {})
    remote_value_tensor = context.create_tensor(ov.Type.f16, ov.Shape(tuple(values.shape)), {})
    remote_pos_tdst_tensor = context.create_tensor(ov.Type.i64, ov.Shape(tuple(pos_tdst.shape)), {})
    # remote_out_tensor = context.create_tensor(ov.Type.f16, ov.Shape(tuple(queries.shape)), {})

    if device == 'GPU':
        ov.Tensor(queries).copy_to(remote_query_tensor)
        ov.Tensor(keys).copy_to(remote_key_tensor)
        ov.Tensor(values).copy_to(remote_value_tensor)
        ov.Tensor(pos_tdst).copy_to(remote_pos_tdst_tensor)

# print(context, remote_query_tensor, remote_key_tensor, remote_value_tensor, remote_out_tensor)

# query_port = compiled_model.input(0)
# key_port = compiled_model.input(1)
# value_port = compiled_model.input(2)

# output_port = compiled_model.output(0)

if device in  ['GPU', 'NPU']:
    request.set_tensor('Parameter_1', remote_query_tensor)
    request.set_tensor('Parameter_2', remote_key_tensor)
    request.set_tensor('Parameter_3', remote_value_tensor)
    request.set_tensor('Parameter_4', remote_pos_tdst_tensor)
# request.set_tensor('out_node', remote_out_tensor)
# print(type(query_port), key_port, value_port, output_port)

sampled = []
for i in tqdm.tqdm(range(100)):
    start = time.perf_counter()
    output = request.infer(
        inputs=[
            remote_query_tensor, 
            remote_key_tensor, 
            remote_value_tensor,
            remote_pos_tdst_tensor,
        ] if device in ['GPU', 'NPU'] else [
            queries,
            keys,
            values,
            pos_tdst,
        ],
        share_inputs=True,
        share_outputs=True,
        decode_strings=False
    )
    elapsed = (time.perf_counter() - start) * 1000
    if i > 3:
        sampled.append(elapsed)
    predictions = next(iter(output.values()))
    # print(elapsed, predictions[0, :, 0, 0])
print('avg', sum(sampled) / len(sampled))