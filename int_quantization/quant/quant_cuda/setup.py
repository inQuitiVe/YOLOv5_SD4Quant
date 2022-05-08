from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ops_cuda',
    ext_modules=[
        CUDAExtension('quant_cuda', [
            'quant_cuda.cpp',
            'quant.cu',
            'float_kernel.cu',
            'bit_helper.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
