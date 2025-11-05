import torch
import my_extension_cpp


# Define a wrapper function
def add_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.ops.my_extension_cpp.add_tensors_metal.default(a, b)
