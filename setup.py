from setuptools import setup, find_packages, find_namespace_packages
from setuptools.command.build_ext import build_ext
from setuptools import Extension
import subprocess
import os, glob
import shutil
import sysconfig

class CMakeExtension(Extension):
    def __init__(self, name): 
        super().__init__(name, sources=[])

class CMakeBuildExt(build_ext):

  def run(self):
    # a bit of a hacky way to do it
    # Create all necessary directories first
    ember_dirs = [
        os.path.join(self.build_lib, 'ember'),
        os.path.join(self.build_lib, 'ember/aten'),
        os.path.join(self.build_lib, 'ember/datasets'),
        os.path.join(self.build_lib, 'ember/models'),
        os.path.join(self.build_lib, 'ember/models/supervised'),
        os.path.join(self.build_lib, 'ember/models/unsupervised'),
        os.path.join(self.build_lib, 'ember/objectives'),
        os.path.join(self.build_lib, 'ember/optimizers'),
        os.path.join(self.build_lib, 'ember/samplers')
    ]
    
    for directory in ember_dirs:
        os.makedirs(directory, exist_ok=True)
        
    # Now proceed with normal build
    super().run()
    
    # Copy Python files
    src_base = 'ember'
    dest_base = os.path.join(self.build_lib, 'ember')
    
    # Helper function to copy files
    def copy_module_files(src_dir, dest_dir):
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for file in glob.glob(os.path.join(src_dir, '*.py')) + glob.glob(os.path.join(src_dir, '*.pyi')):
            shutil.copy2(file, dest_dir)

    # Copy all module files
    copy_module_files(src_base, dest_base)
    copy_module_files(os.path.join(src_base, 'datasets'), os.path.join(dest_base, 'datasets'))
    copy_module_files(os.path.join(src_base, 'models'), os.path.join(dest_base, 'models'))
    copy_module_files(os.path.join(src_base, 'models/supervised'), os.path.join(dest_base, 'models/supervised'))
    copy_module_files(os.path.join(src_base, 'models/unsupervised'), os.path.join(dest_base, 'models/unsupervised'))
    copy_module_files(os.path.join(src_base, 'objectives'), os.path.join(dest_base, 'objectives'))
    copy_module_files(os.path.join(src_base, 'optimizers'), os.path.join(dest_base, 'optimizers'))
    copy_module_files(os.path.join(src_base, 'samplers'), os.path.join(dest_base, 'samplers'))

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

package_data = {
    "ember": ["*.so", "*.pyi", "*.py"],
    "ember.aten": ["*.pyi"],
    "ember.datasets": ["*.py"],
    "ember.models": ["*.py", "supervised/*.py", "unsupervised/*.py"],
    "ember.objectives": ["*.py"],
    "ember.optimizers": ["*.py"],
    "ember.samplers": ["*.py"]
}

packages = find_packages(include=['ember*'])

setup(
    packages=packages,
    package_data=package_data,
    include_package_data=True,
    data_files=[
        ('ember', ['ember/__init__.py']),
        ('ember/aten', ['ember/aten/__init__.pyi']),
        ('ember/datasets', glob.glob('ember/datasets/*.py')),
        ('ember/models', glob.glob('ember/models/*.py')),
        ('ember/models/supervised', glob.glob('ember/models/supervised/*.py')),
        ('ember/models/unsupervised', glob.glob('ember/models/unsupervised/*.py')),
        ('ember/objectives', glob.glob('ember/objectives/*.py')),
        ('ember/optimizers', glob.glob('ember/optimizers/*.py')),
        ('ember/samplers', glob.glob('ember/samplers/*.py'))
    ],
    ext_modules=[CMakeExtension('ember.aten')],
    cmdclass={'build_ext': CMakeBuildExt},
    zip_safe=False
)
