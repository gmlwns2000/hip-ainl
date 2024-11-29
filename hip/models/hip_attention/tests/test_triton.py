import triton
import triton.language as tl
import torch

if torch.xpu.is_available():
    device = 'xpu'
else:
    device = 'cuda'

x = torch.zeros((42, ),).to(device)

@triton.jit
def kernel(A):
    pid = tl.program_id(0)
    tl.store(A + pid, pid)

print('before kernel')
kernel[(len(x),)](x)
print('kernel called')
torch.xpu.synchronize()
print('after kernel')
print(x.cpu())