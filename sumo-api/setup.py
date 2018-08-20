"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


Build and Installation Script (see README.md) for usage
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import os
import sys
import subprocess


args = sys.argv[1:]

from distutils.sysconfig import get_config_vars

# Make a `cleanall` rule to get rid of intermediate and library files
if "cleanall" in args:
    print("Remove build dir if it exists...")
    subprocess.Popen("rm -rf build", shell=True, executable="/bin/bash")
    print("Deleting .so .c and files...")
    subprocess.Popen("find sumo -type f -name '*.so'", shell=True, executable="/bin/bash")
    subprocess.Popen("find sumo -type f -name '*.c'", shell=True, executable="/bin/bash")
    subprocess.Popen("find sumo -type f -name '*.so' -delete", shell=True, executable="/bin/bash")
    subprocess.Popen("find sumo -type f -name '*.c' -delete", shell=True, executable="/bin/bash")

    # Now do a normal clean
    sys.argv[1] = "clean"

# Always use build_ext --inplace
if args.count("build_ext") > 0 and args.count("--inplace") == 0:
    sys.argv.insert(sys.argv.index("build_ext")+1, "--inplace")


# Add new extensions here
extensions = [
    Extension("sumo/base/*",
              sources=["sumo/base/*.pyx"],
              include_dirs=[numpy.get_include(),
                            ".",
                            "/usr/local/include/eigen3",
                            "/usr/include/eigen3"],
              language="c++",
              extra_compile_args=["-std=c++14"],
              extra_link_args=["-std=c++14"]),

    Extension("sumo/geometry/*",
              sources=["sumo/geometry/*.pyx"],
              library_dirs =["/usr/local/Cellar/opencv/3.4.1/lib",
                             '/usr/local/Cellar/boost/1.66.0/lib/'],
              include_dirs=[numpy.get_include(),
                            ".",
                            "/usr/local/include/eigen3",
                            "/usr/include/eigen3",
                            "/usr/local/Cellar/opencv/3.4.1/include",
                            '/usr/local/Cellar/boost/1.66.0/include'],
              language="c++",
              libraries = ['opencv_core', 'boost_system'],
              extra_compile_args=["-std=c++14"],
              extra_link_args=["-std=c++14"]),

    Extension("sumo/opencv/*",
              sources=["sumo/opencv/*.pyx"],
              library_dirs =["/usr/local/Cellar/opencv/3.4.1/lib",
                            '/usr/local/Cellar/boost/1.66.0/lib/'],
              include_dirs=[numpy.get_include(),
                            ".",
                            "/usr/local/include/eigen3",
                            "/usr/include/eigen3",
                            "/usr/local/Cellar/opencv/3.4.1/include",
                            '/usr/local/Cellar/boost/1.66.0/include'],
              libraries = ['opencv_core', 'boost_system'],
              language="c++",
              extra_compile_args=["-std=c++14"],
              extra_link_args=["-std=c++14"]),

    Extension("sumo/images/*",
              sources=["sumo/images/*.pyx",
                       "sumo/images/RgbdTiff.cpp"],
              library_dirs =["/usr/local/Cellar/opencv/3.4.1/lib",
                            '/usr/local/Cellar/boost/1.66.0/lib/'],
              include_dirs=[numpy.get_include(),
                            ".",
                            "/usr/local/include/eigen3",
                            "/usr/include/eigen3",
                            "/usr/local/Cellar/opencv/3.4.1/include",
                            '/usr/local/Cellar/boost/1.66.0/include'],
              libraries = ['tiff', 'opencv_core', 'boost_system'],
              language="c++",
              extra_compile_args=["-std=c++14", "-DPICOJSON_USE_INT64"],
              extra_link_args=["-std=c++14"]),

    
    Extension("sumo/threedee/*",
              sources=["sumo/threedee/*.pyx",
                       "sumo/threedee/Mesh.cpp",
                       "sumo/threedee/TexturedMesh.cpp",
                       "sumo/threedee/GltfModel.cpp"],
              library_dirs =["/usr/local/Cellar/opencv/3.4.1/lib",
                            '/usr/local/Cellar/boost/1.66.0/lib/'],
              include_dirs=[numpy.get_include(),
                            ".",
                            "/usr/local/include/eigen3",
                            "/usr/include/eigen3",
                            "/usr/local/Cellar/opencv/3.4.1/include",
                            '/usr/local/Cellar/boost/1.66.0/include'],
              libraries = ['opencv_core', 'opencv_highgui', 'opencv_imgproc', 'boost_system'],
              language="c++",
              extra_compile_args=["-std=c++14"],
              extra_link_args=["-std=c++14"]),

    # Extension("sumo/semantic/*",
    #           sources=["sumo/semantic/*.pyx",],
    #           library_dirs =["/usr/local/Cellar/opencv/3.4.1/lib",
    #                         '/usr/local/Cellar/boost/1.66.0/lib/'],
    #           include_dirs=[numpy.get_include(),
    #                         ".",
    #                         "/usr/local/include/eigen3",
    #                         "/usr/include/eigen3",
    #                         "/usr/local/Cellar/opencv/3.4.1/include",
    #                         '/usr/local/Cellar/boost/1.66.0/include'],
    #           libraries = ['opencv_core', 'boost_system'],
    #           language="c++",
    #           extra_compile_args=["-std=c++14"],
    #           extra_link_args=["-std=c++14"]),
]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules = cythonize(extensions)
)
