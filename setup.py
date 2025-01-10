from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools import Extension
import subprocess
import os
import shutil
import sysconfig

package_name = "ember"
version = "0.0.1"
description = "Lightweight ML library for my personal use."
author = "Muchang Bahng"
author_email = "bahngmc@gmail.com"
install_requires = [
    "pybind11>=2.10.0",
]

class CMakeExtension(Extension):
    def __init__(self, name):  # Fixed initialization
        super().__init__(name, sources=[])

class CMakeBuildExt(build_ext):

    def build_extension(self, ext):
        # Create build directory if it doesn't exist
        build_dir = os.path.join("build")
        os.makedirs(build_dir, exist_ok=True)

        # Configure CMake
        cmake_args = [
            '-DBUILD_PYTHON_BINDINGS=ON',
        ]

        print("Building extension...")
        # runs cmake for aten to construct the Makefile
        subprocess.check_call(['cmake', '-B', build_dir] + cmake_args)
        # this runs make, which builds main, test, and the .so file  
        subprocess.check_call(['cmake', '--build', build_dir, '--config', 'Release'])

        # This is where CMake puts it
        built_so = os.path.join("build", f"aten{sysconfig.get_config_var('EXT_SUFFIX')}") 
        if not os.path.exists(built_so):
            raise RuntimeError(f"Build failed: {built_so} not found!")

        # Get the target location from setuptools
        ext_fullpath = self.get_ext_fullpath(ext.name) 
        os.makedirs(os.path.dirname(ext_fullpath), exist_ok=True)
        
        shutil.move(built_so, ext_fullpath)
        print(f"Successfully {built_so} to {ext_fullpath}")
setup(
    name=package_name,
    version=version,
    description=description,
    author=author,
    author_email=author_email,
    # build/ext_modules is where we move the .so to both 
    # in the build directory and the source directory 
    ext_modules=[CMakeExtension('ember.aten')], 
    cmdclass={"build_ext": CMakeBuildExt},
    zip_safe=False,
    packages=find_packages(),
    python_requires=">= 3.12.0",
    install_requires=install_requires,
    setup_requires=["pybind11>=2.10.0"]
)
