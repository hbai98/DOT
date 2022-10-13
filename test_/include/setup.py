from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lltm_cpp',
    ext_modules=[
        CUDAExtension('test.csrc', ['test.cu']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
