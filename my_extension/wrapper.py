from importlib import resources

import torch
import my_extension_cpp


# Define a wrapper function
def add_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    # Find the shader file path
    # with resources.as_file(resources.files(__package__) / "shaders" / "add_tensors.metal") as metallib_path:
    #     shader_file_path = str(metallib_path)

    with resources.as_file(resources.files(__package__) / "metal" / "default.metallib") as metallib_path:
        shader_binary_path = str(metallib_path)

    # Call the C++ function
    return torch.ops.my_extension_cpp.add_tensors_metal.default(a, b, shader_binary_path)