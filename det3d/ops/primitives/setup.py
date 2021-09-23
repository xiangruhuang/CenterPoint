from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='primitives',
    ext_modules=[
        CppExtension('primitives_cpu', [
            'src/primitives_cpu.cpp',
            'src/primitives_api.cpp',
        ],
        extra_compile_args={'cxx': ['-g', '-I /usr/local/cuda/include']})
    ],
    cmdclass={'build_ext': BuildExtension})
