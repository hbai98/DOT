from setuptools import setup
import os
import os.path as osp
import warnings

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))

__version__ = None
exec(open('adsvox/version.py', 'r').read())

CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []
include_dirs = [osp.join(ROOT_DIR, "adsvox", "csrc", "include")]

# From PyTorch3D
cub_home = os.environ.get("CUB_HOME", None)
if cub_home is None:
	prefix = os.environ.get("CONDA_PREFIX", None)
	if prefix is not None and os.path.isdir(prefix + "/include/cub"):
		cub_home = prefix + "/include"

if cub_home is None:
	warnings.warn(
		"The environment variable `CUB_HOME` was not found."
		"Installation will fail if your system CUDA toolkit version is less than 11."
		"NVIDIA CUB can be downloaded "
		"from `https://github.com/NVIDIA/cub/releases`. You can unpack "
		"it to a location of your choice and set the environment variable "
		"`CUB_HOME` to the folder containing the `CMakeListst.txt` file."
	)
else:
	include_dirs.append(os.path.realpath(cub_home).replace("\\ ", " "))

try:
    ext_modules = [
        CUDAExtension('adsvox.csrc', [
            'adsvox/csrc/adsvox.cpp',
            'adsvox/csrc/kernel.cu',
            'adsvox/csrc/rt_kernel.cu',
            'adsvox/csrc/quantizer.cpp',
        ], include_dirs=include_dirs,
        optional=False),
    ]
except:
    import warnings
    warnings.warn("Failed to build CUDA extension")
    ext_modules = []

setup(
    name='adsvox',
    version=__version__,
    author='Haotian Bai',
    author_email='haotianwhite@outlook.com',
    description='Adaptive PyTorch sparse voxel volume extension, including custom CUDA kernels',
    url='https://github.com/164140757/AdaptiveNerf',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    packages=['adsvox', 'adsvox.csrc'],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)

