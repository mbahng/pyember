from setuptools import setup, find_packages
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

package_name = "ember" 
version = "0.0.1" 
description = "Lightweight ML library for my personal use."
long_description = "To be done. " 

author = "Muchang Bahng"
author_email = "bahngmc@gmail.com"

install_requires = []


# Check if we're on macOS ARM64
is_macos_arm64 = sys.platform == "darwin" and "arm" in sys.version.lower()

# Compiler and linker flags
extra_compile_args = ['-std=c++11']
extra_link_args = []

if is_macos_arm64:
    extra_compile_args.extend(['-arch', 'arm64'])
    extra_link_args.extend(['-arch', 'arm64'])

ext_modules = [
    Pybind11Extension(
        "ember/tensor_cpp",
        ["ember/tensor.cpp", "ember/tensor_binding.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name = package_name, 
    version = version, 
    description = description, 
    long_description = long_description, 
    author = author, 
    author_email = author_email, 
    ext_modules = ext_modules,
    cmdclass = {"build_ext" : build_ext}, 
    zip_safe = False,
    packages = find_packages(), 
    python_requires = ">= 3.7", 
    install_requires = install_requires
)


