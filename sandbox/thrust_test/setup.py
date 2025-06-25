"""
This file demonstrates how to set up a CMake-based build for a Python extension
that uses Thrust/CUDA and can be integrated with your QForte project.
"""

import os
import sys
import subprocess
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Get the directory of this script
this_dir = os.path.dirname(os.path.abspath(__file__))

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the extension")
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Set build type
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]
        
        # Set build type
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        
        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']
        
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version()
        )
        
        # Build directory
        build_temp = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)
        
        # Build the extension
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)

# Setup the Python module
setup(
    name='thrust_test_py',
    version='0.1',
    author='QForte Developer',
    author_email='qforte@example.com',
    description='Test of Thrust/CUDA integration with Python',
    long_description='',
    ext_modules=[CMakeExtension('thrust_test_py')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
