"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import cv2
import numpy as np
cimport numpy as np
from libc.string cimport memcpy
from libc.stdint cimport uint16_t


cdef Mat* mat_from_array(np.ndarray array):
  """ Convert numpy array to generic OpenCV Mat.
      Keyword arguments:
          array -- numpy array, 3-channel uint8 or 1-channel uint8 or float32
      We do not handle 1-D arrays, just images.
      Returns a pointer to a C++ Mat object, owned by OpenCV.
  """
  assert <unsigned char*>array.data != <unsigned char*>0
  cdef Mat* mat
  if array.ndim == 2:
    # Handle grayscale images
    if array.dtype == np.uint8:
      mat = new Mat(array.shape[0], array.shape[1], cv2.CV_8UC1,
                    <unsigned char*>array.data, array.strides[0])
    else:
      assert array.dtype == np.float32, 'mat_from_array needs uint8 or float32.'
      mat = new Mat(array.shape[0], array.shape[1], cv2.CV_32FC1,
                    <unsigned char*>array.data, array.strides[0])
  else:
      # Handle color images
      assert array.ndim == 3, 'mat_from_array needs 2 or 3-dim array.'
      assert array.dtype == np.uint8, 'mat_from_array needs uint8.'
      mat = new Mat(array.shape[0], array.shape[1], cv2.CV_8UC3,
                    <unsigned char*>array.data, array.strides[0])
  if mat == <Mat*>0:
      raise ValueError('mat_from_array failed')
  return mat

cdef Mat1b* mat1b_from_array(np.ndarray[np.uint8_t, ndim=2, mode='c'] array):
  assert <unsigned char*>array.data != <unsigned char*>0
  cdef Mat1b* mat = new Mat1b(
      array.shape[0], array.shape[1], <unsigned char*>array.data, array.strides[0])
  if mat == <Mat1b*>0:
      raise ValueError('mat1b_from_array failed')
  return mat

cdef Mat1w* mat1w_from_array(np.ndarray[np.uint16_t, ndim=2, mode='c'] array):
  assert <unsigned short*>array.data != <unsigned short*>0
  return new Mat1w(
      array.shape[0], array.shape[1], <unsigned short*>array.data, array.strides[0])

cdef Mat1f* mat1f_from_array(np.ndarray[np.float32_t, ndim=2, mode='c'] array):
  assert <unsigned char*>array.data != <unsigned char*>0
  cdef Mat1f* mat = new Mat1f(
      array.shape[0], array.shape[1], <np.float32_t*>array.data, array.strides[0])
  if mat == <Mat1f*>0:
      raise ValueError('mat1f_from_array failed')
  return mat

cdef Mat3b* mat3b_from_array(np.ndarray[np.uint8_t, ndim=3, mode='c'] array):
  assert <Vec3b*>array.data != <Vec3b*>0
  cdef Mat3b* mat = new Mat3b(
      array.shape[0], array.shape[1], <Vec3b*>array.data, array.strides[0])
  if mat == <Mat3b*>0:
      raise ValueError('mat3b_from_array failed')
  return mat

cdef array_from_mat(Mat mat):
  """ Convert generic OpenCV Mat to numpy array.
      Keyword arguments:
          mat -- pointer to a C++ Mat object, type CV_8UC1, CV_8UC1 or CV_32FC1
      Returns a numpy array.
  """
  cdef np.ndarray a
  cdef size_t r = mat.rows, c = mat.cols # image size
  cdef size_t n = mat.channels(), d = 1 # channels and depth (in bytes)
  if n==1:
      # Handle grayscale images
      if mat.type() == cv2.CV_8UC1:
          a = np.empty((r,c), dtype=np.uint8)
      else:
          assert mat.type() == cv2.CV_32FC1, 'array_from_mat needs uint8 or float32.'
          a = np.empty((r,c), dtype=np.float32)
          d = 4
  else:
      # Handle color images
      assert n == 3, 'array_from_mat needs 1 or 3-channel Mat, not {}.'.format(n)
      assert mat.type() == cv2.CV_8UC3, 'array_from_mat needs uint8.'
      a = np.empty((r,c,3), dtype=np.uint8)
  if mat.data != <void*>0:
      memcpy(a.data, mat.data, r * c * n * d)
  return a

cdef array_from_mat1b(Mat1b mat):
  cdef int r = mat.rows
  cdef int c = mat.cols
  cdef np.ndarray[np.uint8_t, ndim=2, mode = 'c'] a = np.empty((r, c), dtype=np.uint8)
  if mat.data != <void*>0:
      memcpy(a.data, mat.data, r * c)
  return a

cdef array_from_mat1w(Mat1w mat):
  cdef int r = mat.rows
  cdef int c = mat.cols
  cdef np.ndarray[np.uint16_t, ndim=2, mode = 'c'] a = np.empty((r, c), dtype=np.uint16)
  if mat.data != <void*>0:
      memcpy(a.data, mat.data, r * c * sizeof(uint16_t))
  return a

cdef array_from_mat1f(Mat1f mat):
  cdef int r = mat.rows
  cdef int c = mat.cols
  cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] a = np.empty((r, c), dtype=np.float32)
  if mat.data != <void*>0:
      memcpy(a.data, mat.data, r * c * sizeof(np.float32_t))
  return a

cdef array_from_mat3b(Mat3b mat):
  cdef int r = mat.rows
  cdef int c = mat.cols
  cdef np.ndarray[np.uint8_t, ndim=3, mode = 'c'] a = np.empty((r, c, 3), dtype=np.uint8)
  if mat.data != <void*>0:
      memcpy(a.data, mat.data, r * c * 3)
  return a
