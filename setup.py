from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pytorch_noise',
    version = '0.0.2',
    author = "Dylan McLeod",
    author_email = "mcleoddylan0121@gmail.com",
    ext_modules=[
        CUDAExtension('pytorch_noise_cuda', [
            'cuda/bindings.cpp',
            'cuda/kernels.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=["pytorch_noise"]
)