from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools import Extension
import subprocess
import os, glob
import shutil
import sysconfig

class CMakeExtension(Extension):
    def __init__(self, name):  # Fixed initialization
        super().__init__(name, sources=[])

class CMakeBuildExt(build_ext):
  def build_extension(self, ext):
    # remove all previous build directories and .so files if they exist
    for so_file in glob.glob(os.path.join("ember", "*.so")): 
      os.remove(so_file)
    if os.path.isdir(os.path.join("build")): 
      shutil.rmtree(os.path.join("build"))

    # Create build directory if it doesn't exist
    build_dir = os.path.join("build")
    os.makedirs(build_dir, exist_ok=True)
    debug = os.environ.get('CMAKE_DEBUG', '').upper() in ('1', 'ON', 'TRUE', 'YES')
    dev = os.environ.get('CMAKE_DEV', '').upper() in ('1', 'ON', 'TRUE', 'YES')

    # Configure CMake
    cmake_args = [ 
                  '-B', build_dir, 
                  '-S', os.path.join(os.path.dirname(__file__), 'aten'),
                  '-DBUILD_PYTHON_BINDINGS=ON', 
                  f'-DBUILD_DEBUG={debug}', 
                  f'-DBUILD_DEV={dev}'
    ]
    print("Building extension...")
    # runs cmake for aten to construct the Makefile
    subprocess.check_call(['cmake'] + cmake_args)
    # this runs make, which builds main, test, and the .so file  
    subprocess.check_call(['cmake', '--build', build_dir, '--config', 'Release'])

    # This is where CMake puts it
    if os.name == 'nt':  # Windows
        built_so = os.path.join(build_dir, 'Release', f"aten{sysconfig.get_config_var('EXT_SUFFIX')}") 
    else:  # Linux/Mac
        built_so = os.path.join(build_dir, f"aten{sysconfig.get_config_var('EXT_SUFFIX')}")
    
    if not os.path.exists(built_so):
      raise RuntimeError(f"Build failed: {built_so} not found!")

    # Get the target location from setuptools
    ext_fullpath = self.get_ext_fullpath(ext.name) 
    os.makedirs(os.path.dirname(ext_fullpath), exist_ok=True)
    shutil.copy(built_so, ext_fullpath)
    
    print(f"Successfully moved {built_so} to {ext_fullpath}")
    
    # the .so file should be moved from ext_fullpath to ember/, but on ubuntu it doesn't 
    # so we move it manually 
    shutil.move(built_so, "ember")
    print(f"Successfully moved {built_so} to ember/")

packages = find_packages() 
packages.extend("ember.aten")
print("===================")
for p in packages: 
  print(p)
print("===================")

ember_package_data = [
  "ember/*", 
  "ember/aten/*",
  "ember/datasets/*", 
  "ember/models/*", 
  "ember/models/supervised/*", 
  "ember/models/unsupervised/*", 
  "ember/objectives/*", 
  "ember/optimizers/*", 
  "ember/samplers/*"
]
package_data = {
  "ember" : ember_package_data
}

setup( 
  packages = packages,
  package_data = package_data, 
  ext_modules = [CMakeExtension('ember.aten')], 
  cmdclass={'build_ext': CMakeBuildExt}
)
