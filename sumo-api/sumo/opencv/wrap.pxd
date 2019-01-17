"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Imports opencv types and declares conversion functions to/from numpy.
"""

import numpy as np
cimport numpy as np

from libcpp cimport bool

cdef extern from "<opencv2/core/core.hpp>":
  cdef int CV_8UC3
  cdef int CV_8UC1
  cdef int CV_16UC1

cdef extern from "<opencv2/core/core.hpp>" namespace "cv":
  cdef cppclass Size:
    Size()
    Size(size_t, size_t)

  # Generic image
  cdef cppclass Mat:
    Mat(int rows, int cols, int type, unsigned char* data, size_t step) except +
    Mat() except +
    int channels()
    int type()
    int rows, cols
    void* data
    bool empty()

  # Grayscale image (uint8)
  cdef cppclass Mat1b:
    Mat1b(int rows, int cols, unsigned char* data, size_t step) except +
    Mat1b() except +
    void create(int rows, int cols)
    int rows, cols
    void* data

  # Grayscale image (uint16)
  cdef cppclass Mat1w:
    Mat1w(int rows, int cols, unsigned short* data, size_t step) except +
    Mat1w() except +
    void create(int rows, int cols)
    int rows, cols
    void* data

  # Grayscale image, float32
  cdef cppclass Mat1f:
    Mat1f(int rows, int cols, float* data, size_t step) except +
    Mat1f() except +
    void create(int rows, int cols)
    int rows, cols
    void* data

  # RGB value
  cdef cppclass Vec3b:
      pass

  # RGB image
  cdef cppclass Mat3b:
    Mat3b(int rows, int cols, Vec3b* data, size_t step) except +
    Mat3b() except +
    void create(int rows, int cols)
    int rows, cols
    void* data

# Convert from numpy to opencv
# NOTE: these do not copy memory, but provide "views" onto the numpy data.
cdef Mat* mat_from_array(np.ndarray array) except +
cdef Mat1b* mat1b_from_array(np.ndarray[np.uint8_t, ndim=2, mode='c'] array) except +
cdef Mat1w* mat1w_from_array(np.ndarray[np.uint16_t, ndim=2, mode='c'] array) except +
cdef Mat1f* mat1f_from_array(np.ndarray[np.float32_t, ndim=2, mode='c'] array) except +
cdef Mat3b* mat3b_from_array(np.ndarray[np.uint8_t, ndim=3, mode='c'] array) except +

# Convert from opencv to numpy
# NOTE: these *do* copy data, i.e., they assume OpenCV owns the memory buffer.
cdef array_from_mat(Mat mat) except +
cdef array_from_mat1b(Mat1b mat) except +
cdef array_from_mat1w(Mat1w mat) except +
cdef array_from_mat1f(Mat1f mat) except +
cdef array_from_mat3b(Mat3b mat) except +
