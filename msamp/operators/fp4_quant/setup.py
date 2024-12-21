# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# added ruizhe 2024/12/20

"""The setuptools based setup module."""

from setuptools import setup
from torch.utils import cpp_extension

ext_t = cpp_extension.CUDAExtension
ext_fnames = ['my_quant.cu']
define_macros = []
extra_compile_args = dict(cxx=['-fopenmp', '-O3'], nvcc=['-O3'])

define_macros.append(('WITH_CUDA', None))

setup(
    name='msamp_my_quant',
    version='0.0.1',
    ext_modules=[
        ext_t(
            'msamp_my_quant',
            ext_fnames,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)