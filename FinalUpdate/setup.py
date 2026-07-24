from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "sampling",
        ["sources/sampling.cpp"],
        extra_compile_args=['-std=c++11'],  # 可选编译选项
    ),
]

setup(
    name="sampling",
    version="1.0",
    ext_modules=ext_modules,
)