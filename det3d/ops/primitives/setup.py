from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='primitives',
    ext_modules=[
        CppExtension('primitives_cuda', [
            'src/primitives_cpu.cpp',
            'src/primitives_api.cpp',
            'src/primitives_hash.cpp',
            'src/primitives_hash.cu',
        ],
        extra_compile_args={'cxx': ['-g', '-I /usr/local/cuda/include'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
