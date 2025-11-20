# setup.py
import numpy as np
from setuptools import setup, find_packages, Extension

__version__ = "0.0.1"
CFLAGS = ["-Ofast"]

ext_modules = [
    Extension(
        name="fluidgpu._solver",
        sources=["fluidgpu/_solver.c"],
        extra_compile_args=CFLAGS,
        include_dirs=[
            np.get_include(),  # numpy headers
            # "fluidgpu",      # if more header files are added we can include whole directory
        ],
    )
]

setup(
    name="fluidgpu",
    version=__version__,
    packages=find_packages(),
    ext_modules=ext_modules,
)
