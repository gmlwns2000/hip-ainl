import openvino as ov
import torch

@torch.jit.script
def target(x: torch.Tensor):
    # x = x.repeat_interleave(10, dim=-1)
    # return x
    x.mul_(42)
    return x

model = ov.convert_model(target, input=[(10, 32)], verbose=True)
ov.save_model(model, 'test.xml')