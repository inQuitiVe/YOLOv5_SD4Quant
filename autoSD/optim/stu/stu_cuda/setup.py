from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='autoSD_cuda',
    ext_modules=[
        CUDAExtension('stu_cuda', [
            'stu_cuda.cpp',
            'stu.cu',
            'stu_kernel.cu',
            'stu_helper.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
