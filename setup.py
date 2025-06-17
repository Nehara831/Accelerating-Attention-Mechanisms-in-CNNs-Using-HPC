from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension, find_packages
import os

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "attention_cuda_py",
        [
            "src/cpp/python_binding.cpp",
            "src/cpp/attention.cpp",
            "src/cpp/attention_cuda.cu",
        ],
        include_dirs=[
            pybind11.get_include(),
            "/usr/local/cuda/include",
            "src/cpp"
        ],
        libraries=["cudart"],
        library_dirs=["/usr/local/cuda/lib64"],
        language='c++',
        cxx_std=14,
    ),
]

setup(
    name="attention_cnn",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.60.0",
    ],
)