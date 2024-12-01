import torch
import openvino as ov
import openvino.properties as props
import numpy as np

# @torch.jit.script
# def test(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
#     x = A @ B
#     return x

class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        x = torch.zeros((1, ))
        return x
test = TestModel()

linear = torch.nn.Linear(4096, 4096).half().eval()

inC, outC, batch = 4096, 4096, 32
X1 = torch.tensor(np.random.uniform(-1, 1, (batch, 16, inC)).astype(np.float16))
X2 = torch.tensor(np.random.uniform(-1, 1, (inC, outC)).astype(np.float16))

ov_model = ov.convert_model(linear, verbose=True, example_input=(X1,), share_weights=False)

print(ov_model)

core = ov.Core()

devices = core.available_devices
for device in devices:
    device_name = core.get_property(device, props.device.full_name)
    print(f"{device}: {device_name}")

device = 'NPU'

print(f"{device} SUPPORTED_PROPERTIES:\n")
supported_properties = core.get_property(device, props.supported_properties)
indent = len(max(supported_properties, key=len))

for property_key in supported_properties:
    if property_key not in ("SUPPORTED_METRICS", "SUPPORTED_CONFIG_KEYS", "SUPPORTED_PROPERTIES"):
        try:
            property_val = core.get_property(device, property_key)
        except TypeError:
            property_val = "UNSUPPORTED TYPE"
        print(f"{property_key:<{indent}}: {property_val}")

compiled_model = core.compile_model(ov_model, 'NPU', )
input('>>>')
