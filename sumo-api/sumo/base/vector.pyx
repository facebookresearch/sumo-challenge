"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Some utilities for small vectors.

    We use numpy vectors for small vectors, but they are common enough
    to give them constructor-like creation functions, as if they were classes.

    Below that we also define the conversion functions to/from Eigen.
"""

import numpy as np
cimport numpy as np
from libc.string cimport memcpy

cpdef np.ndarray Vector2(double x, double y):
    """Constructor-like creation of a float 2-vector."""
    return np.array([x, y], dtype=np.float64)

cpdef np.ndarray Vector2f(double x, double y):
    """Constructor-like creation of a float32 2-vector."""
    return np.array([x, y], dtype=np.float32)

cpdef np.ndarray Vector3(double x, double y, double z):
    """Constructor-like creation of a float 3-vector."""
    return np.array([x, y, z], dtype=np.float64)

cpdef np.ndarray Vector3f(double x, double y, double z):
    """Constructor-like creation of a float32 3-vector."""
    return np.array([x, y, z], dtype=np.float32)

cpdef np.ndarray Vector4(double x, double y, double z, double w):
    """Constructor-like creation of a float 4-vector."""
    return np.array([x, y, z, w], dtype=np.float64)

cpdef np.ndarray Matrix3(x, y, z):
    """Constructor-like creation of  3*3-matrix."""
    return np.array([x, y, z], dtype=np.float64)


# Vector2 conversions

cdef CVector2 vector2(np.ndarray a):
  return CVector2(a[0],a[1])

cdef np.ndarray array2(CVector2 v):
  return np.array([v.x(), v.y()], dtype=np.float64)

# Vector2f conversions

cdef CVector2f vector2f(np.ndarray a):
  return CVector2f(a[0],a[1])

cdef np.ndarray array2f(CVector2f v):
  return np.array([v.x(), v.y()], dtype=np.float32)

cdef vector[CVector2f] vector2fs_of_array(np.ndarray points):
  assert points.shape[0] == 2
  cdef vector[CVector2f] cpp_points
  cdef CVector2f cpp_point
  for i in range(points.shape[1]):
      cpp_point = vector2f(points[:,i])
      cpp_points.push_back(cpp_point)
  return cpp_points

cdef np.ndarray array_of_vector2fs(vector[CVector2f] cpp_points):
  cdef size_t n = cpp_points.size()
  cdef np.ndarray points = np.empty((2,n), dtype=np.float32)
  for i in range(n):
    points[:,i] = array2f(cpp_points[i])
  return points

# Vector3 conversions

cdef CVector3 vector3(np.ndarray a):
  return CVector3(a[0],a[1],a[2])

cdef np.ndarray array3(CVector3 v):
  return np.array([v.x(), v.y(), v.z()], dtype=np.float64)

cdef vector[CVector3] vector3s_of_array(np.ndarray points):
  assert points.shape[0] == 3
  cdef vector[CVector3] cpp_points
  cdef CVector3 cpp_point
  for i in range(points.shape[1]):
      cpp_point = vector3(points[:,i])
      cpp_points.push_back(cpp_point)
  return cpp_points

cdef np.ndarray array_of_vector3s(vector[CVector3] cpp_points):
  cdef size_t n = cpp_points.size()
  points = np.empty((3,n), dtype=np.float)
  for i in range(n):
    points[:,i] = array3(cpp_points[i])
  return points

# Vector3f conversions

cdef CVector3f vector3f(np.ndarray a):
  return CVector3f(a[0],a[1],a[2])

cdef np.ndarray array3f(CVector3f v):
  return np.array([v.x(), v.y(), v.z()], dtype=np.float32)

cdef vector[CVector3f] vector3fs_of_array(np.ndarray points):
  assert points.shape[0] == 3
  cdef vector[CVector3f] cpp_points
  cdef CVector3f cpp_point
  for i in range(points.shape[1]):
      cpp_point = vector3f(points[:,i])
      cpp_points.push_back(cpp_point)
  return cpp_points

cdef np.ndarray array_of_vector3fs(vector[CVector3f] cpp_points):
  cdef size_t n = cpp_points.size()
  cdef np.ndarray points = np.empty((3,n), dtype=np.float32)
  for i in range(n):
    points[:,i] = array3f(cpp_points[i])
  return points

# Vector4 conversions

cdef CVector4 vector4(np.ndarray a):
  return CVector4(a[0],a[1],a[2],a[3])

cdef np.ndarray array4(CVector4 v):
  return np.array([v.x(), v.y(), v.z(), v.w()], dtype=np.float64)

def unitize(v):
    """
    Convert numpy vector <v> into a unit vector and return.
    If <v> is the 0 vector, then <v> is returned.
    """
    norm = np.linalg.norm(v)
    if (norm != 0):
        return v / np.linalg.norm(v)
    else:
        return v

def on_left(N, p, a, b):
    """
    Returns true iff point <p> is on left side of line b-a
    as seen looking along the -N axis

    Inputs:
    N (numpy 3x1 vector) - normal
    p (numpy 3x1 vector) - query point
    a, b (numpy 3x1 vectors) - 2 points on query line

    Return:
    Boolean
    """
    return np.dot(N, np.cross(b-a, p-a)) >= 0

cdef array_from_matrix3(CMatrix3 mat):
    # order is chosen to be 'F' (Fortran) in order that the data storing order
    # is column-major
    cdef np.ndarray[np.double_t, ndim=2] a = np.empty((3, 3), dtype=np.double, order='F')
    memcpy(a.data, mat.data(), 9 * sizeof(np.double_t))
    return a

cdef CMatrix3 matrix3_of_array(np.ndarray a):
    cdef CMatrix3 mat
    cdef np.ndarray[np.double_t, ndim=2] f = np.asfortranarray(a)
    memcpy(mat.data(), f.data, 9 * sizeof(np.double_t))
    return mat
