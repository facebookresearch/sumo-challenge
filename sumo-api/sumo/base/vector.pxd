"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Wrap small Eigen types defined in Vector.h, along with conversions
"""

cimport numpy as np
from libcpp.vector cimport vector

# import Eigen vectors from C++
cdef extern from "sumo/base/Vector.h" namespace "sumo":

  cdef cppclass CVector2 "sumo::Vector2":
    CVector2() except +
    CVector2(double,double) except +
    double& operator[](int)
    double x() const
    double y() const
    CVector2 operator+(const CVector2&)
    CVector3 operator*(double)
    CVector3 operator/(double)
    double norm() const

  cdef cppclass CVector2f "sumo::Vector2f":
    CVector2f() except +
    CVector2f(np.float32_t,np.float32_t) except +
    np.float32_t x() const
    np.float32_t y() const

  cdef cppclass CVector3 "sumo::Vector3":
    CVector3() except +
    CVector3(double,double,double) except +
    double& operator[](int)
    double x() const
    double y() const
    double z() const
    CVector3 operator+(const CVector3&)
    CVector3 operator*(double)
    CVector3 operator/(double)
    double norm() const

  cdef cppclass CVector3f "sumo::Vector3f":
    CVector3f() except +
    CVector3f(np.float32_t,np.float32_t,np.float32_t) except +
    np.float32_t x() const
    np.float32_t y() const
    np.float32_t z() const

  cdef cppclass CVector4 "sumo::Vector4":
    CVector4() except +
    CVector4(double,double,double,double) except +
    double& operator[](int)
    double x() const
    double y() const
    double z() const
    double w() const
    CVector4 operator+(const CVector4&)
    CVector4 operator*(double)
    CVector4 operator/(double)
    double norm() const

  cdef cppclass CMatrix3 "sumo::Matrix3":
    CMatrix3() except +
    double* data()

# declare conversions to and from numpy (defined in pyx):

cdef array_from_matrix3(CMatrix3 mat)
cdef CMatrix3 matrix3_of_array(np.ndarray input_a)

# Vector2

cdef CVector2 vector2(np.ndarray[np.float32_t, ndim=2])
cdef np.ndarray[np.float32_t, ndim=2] array2(CVector2)

# Vector2f

cdef CVector2f vector2f(np.ndarray[np.float32_t, ndim=1] a)
cdef np.ndarray[np.float32_t, ndim=1] array2f(CVector2f v)
cdef vector[CVector2f] vector2fs_of_array(np.ndarray[np.float32_t, ndim=2] points)
cdef np.ndarray[np.float32_t, ndim=2] array_of_vector2fs(vector[CVector2f] points)

# Vector3

cdef CVector3 vector3(np.ndarray)
cdef np.ndarray array3(CVector3)
cdef vector[CVector3] vector3s_of_array(np.ndarray)
cdef np.ndarray array_of_vector3s(vector[CVector3])

# Vector3f

cdef CVector3f vector3f(np.ndarray[np.float32_t, ndim=1] a)
cdef np.ndarray[np.float32_t, ndim=1] array3f(CVector3f v)
cdef vector[CVector3f] vector3fs_of_array(np.ndarray[np.float32_t, ndim=2] points)
cdef np.ndarray[np.float32_t, ndim=2] array_of_vector3fs(vector[CVector3f] points)

# Vector4
cdef CVector4 vector4(np.ndarray)
cdef np.ndarray array4(CVector4)
