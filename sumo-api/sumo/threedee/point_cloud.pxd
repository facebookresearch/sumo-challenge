# Copyright 2004-present Facebook. All Rights Reserved.
"""Cython header for interfacing with PointCloud C++."""

cimport numpy as np
from sumo.base.vector cimport CVector3
from libcpp.vector cimport vector
from libcpp cimport bool

# Wrap Point Cloud C++ class
cdef extern from "sumo/threedee/PointCloud.h" namespace "sumo":
  cdef cppclass CColor "sumo::PointCloud::Color":
    unsigned char r
    unsigned char g
    unsigned char b

  cdef cppclass CPointCloud "sumo::PointCloud":
    CPointCloud() except +
    CPointCloud(const CPointCloud&) except +
    CPointCloud(const vector[CVector3]&) except +
    CPointCloud(const vector[CVector3]&, const vector[CColor]&) except +
    CPointCloud(const vector[const CPointCloud*]&) except +
    void append(const CPointCloud& other)
    const vector[CVector3]& points() const
    const CVector3& point(size_t) const
    const vector[CColor]& colors() const
    const CColor& color(size_t) const
    size_t numPoints() const
    bool colored() const

# Python wrapper class storing a pointer:
cdef class PointCloud:
  cdef CPointCloud* _c_ptr
