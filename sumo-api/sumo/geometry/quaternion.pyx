"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import numpy as np
cimport numpy as np

from sumo.base.vector cimport CMatrix3, array_from_matrix3, matrix3_of_array

cdef extern from "<Eigen/Geometry>" namespace "Eigen":

  cdef cppclass Quaterniond:
    Quaterniond(double, double, double, double) except +
    Quaterniond(const CMatrix3&) except +
    double w() const
    double x() const
    double y() const
    double z() const
    Quaterniond normalized() const
    CMatrix3 toRotationMatrix() const
    CMatrix3 matrix() const

cdef class Quaternion:
  """Wrapper for Eigen::Quaterniond."""

  cdef Quaterniond* c_ptr_

  def __cinit__(self, np.ndarray val=np.array([1., 0., 0., 0.])):
    """Constructor, <val> is [1, 0, 0, 0] (w, x, y, z) by default."""
    if val.shape[0]==3:
      assert val.shape[1] == 3, "Quaternion takes 4-vector or 3*3 matrix"
      self.c_ptr_ = new Quaterniond(matrix3_of_array(val))
    else:
      assert val.shape[0] == 4, "Quaternion takes 4-vector or 3*3 matrix"
      self.c_ptr_ = new Quaterniond(val[0], val[1], val[2], val[3])

  def as_vector(self):
    """return quaternion as a vector"""
    q = self.c_ptr_
    return np.array([q.w(), q.x(), q.y(), q.z()])

  def to_rotation_matrix(self):
    """return the 3x3 rotation matrix """
    return array_from_matrix3(self.c_ptr_.normalized().matrix())
