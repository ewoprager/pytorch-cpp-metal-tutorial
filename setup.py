import os
import subprocess

from numpy.distutils.lib2def import output_def
from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import CppExtension, BuildExtension
import pathlib


def compile_metal_shaders(shader_dir: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
    """
    Requires the xcodebuild MetalToolChain to be installed.
    :param shader_dir:
    :param output_dir:
    :return:
    """
    # compile each .metal shader into a .air file
    output_dir.mkdir(exist_ok=True, parents=True)
    air_files = []
    for source in shader_dir.iterdir():
        if source.suffix != ".metal":
            continue
        air = output_dir / source.with_suffix(".air").name
        print("Compiling Metal shader: {}".format(str(source)))
        subprocess.check_call(["xcrun", "-sdk", "macosx", "metal", "-c", str(source), "-o", str(air)])
        air_files.append(str(air))

    # compile air files into a single metallib library
    metallib = output_dir / "default.metallib"
    print("Linking metallib...")
    subprocess.check_call(["xcrun", "-sdk", "macosx", "metallib", *air_files, "-o", str(metallib)])

    # clean up air files
    for air_file in air_files:
        pathlib.Path(air_file).unlink()

    # convert the library into a C-array
    header = output_dir / "default_metallib.h"
    header.write_text(subprocess.run(["xxd", "-i", str(metallib)], capture_output=True, text=True, check=True).stdout)

    # cleaning up library
    metallib.unlink()

    return metallib


def get_extensions():
    # prevent ninja from using too many resources
    try:
        import psutil
        num_cpu = len(psutil.Process().cpu_affinity())
        cpu_use = max(4, num_cpu - 1)
    except (ModuleNotFoundError, AttributeError):
        cpu_use = 4

    os.environ.setdefault('MAX_JOBS', str(cpu_use))

    extra_compile_args = {}
    if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):

        # objc compiler support
        from distutils.unixccompiler import UnixCCompiler
        if '.mm' not in UnixCCompiler.src_extensions:
            UnixCCompiler.src_extensions.append('.mm')
            UnixCCompiler.language_map['.mm'] = 'objc'

        extra_compile_args = {}
        extra_compile_args['cxx'] = ['-Wall', '-std=c++17', '-framework', 'Metal', '-framework', 'Foundation',
                                     '-ObjC++']
    else:
        extra_compile_args['cxx'] = ['-std=c++17']

    ext_ops = CppExtension(  #
        name='my_extension_cpp',  #
        sources=['my_extension/cpp_extension.mm'],  #
        include_dirs=[],  #
        extra_objects=[],  #
        extra_compile_args=extra_compile_args,  #
        library_dirs=[],  #
        libraries=[],  #
        extra_link_args=[]  #
    )
    return [ext_ops]


class BuildExtensionWithMetal(BuildExtension):
    def run(self):
        shader_dir = pathlib.Path("my_extension") / "shaders"
        output_dir = pathlib.Path("my_extension") / "metal"
        if shader_dir.is_dir():
            compile_metal_shaders(shader_dir, output_dir)
        else:
            print("Warning: No shader directory '{}'.".format(str(shader_dir)))
        super().run()


setup(name='my_extension',  #
      version="0.0.1",  #
      packages=find_packages(),  #
      include_package_data=True,  #
      python_requires='>=3.11',  #
      ext_modules=get_extensions(),  #
      cmdclass={'build_ext': BuildExtensionWithMetal},  #
      zip_safe=False,  #
      requires=[  #
          'torch',  #
          'setuptools'  #
      ])
