import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[
    Extension("mask_accelerated",
              ["mask_accelerated.pyx"],
              libraries=["m"],
              extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp']
              )
]

setup(
  name="mask_accelerated",
  cmdclass={"build_ext": build_ext},
  ext_modules=ext_modules
)

# compile instructions:
# python setup.py build_ext --inplace