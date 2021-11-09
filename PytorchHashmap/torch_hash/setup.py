from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='torch_hash',
    ext_modules=[
        CUDAExtension('torch_hash_cuda', [
            'src/torch_hash_api.cpp',
            'src/torch_hash_kernel.cu',
        ],
        extra_compile_args={'cxx': ['-g', '-I /usr/local/cuda/include'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
