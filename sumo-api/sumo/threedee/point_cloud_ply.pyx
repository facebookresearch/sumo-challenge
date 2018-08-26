# Copyright 2004-present Facebook. All Rights Reserved.
import numpy as np
cimport numpy as np

from libcpp.vector cimport vector

cdef np.ndarray numpy_from_unraveled_points(const vector[float]& input):
  cdef int length = input.size()
  assert length % 3 == 0, "Has to be multiple of 3 !!"
  num_of_points = int(length / 3)
  cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] result = \
    np.empty([3, num_of_points], dtype=np.float32)
  cdef int index = 0
  cdef int i = 0
  for i in range(num_of_points):
    result[0,i] = input[index]
    result[1,i] = input[index+1]
    result[2,i] = input[index+2]
    index += 3
  return result

# This is the python wrapper for readPoints
def read_points(str filename):
  py_byte_string = filename.encode('UTF-8')
  cdef char* c_string = py_byte_string
  cdef vector[float] unraveled_points = readPoints(c_string)
  num_of_points = unraveled_points.size() / 3
  return numpy_from_unraveled_points(unraveled_points), num_of_points

# This is the python wrapper for writePoints
def write_points(np.ndarray vertex_points, str filename):
  cdef np.ndarray unraveled_points = np.reshape(
                                                vertex_points,
                                                3*vertex_points.shape[1],
                                                order='F'
                                                )
  py_byte_string = filename.encode('UTF-8')
  cdef char* c_string = py_byte_string
  writePoints(unraveled_points, c_string)

# This is the python wrapper for writePointsAndColors
def write_points_and_colors(np.ndarray vertex_points, np.ndarray vertex_colors, str filename):
  cdef np.ndarray unraveled_points = np.reshape(
                                                vertex_points,
                                                3*vertex_points.shape[1],
                                                order='F'
                                                )
  cdef np.ndarray unraveled_colors = np.reshape(
                                                vertex_colors,
                                                3*vertex_colors.shape[1],
                                                order='F'
                                                )
  py_byte_string = filename.encode('UTF-8')
  cdef char* c_string = py_byte_string
  writePointsAndColors(unraveled_points, unraveled_colors, c_string)
