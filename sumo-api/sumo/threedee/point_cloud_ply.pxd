# Copyright 2004-present Facebook. All Rights Reserved.
from libcpp.vector cimport vector

cdef extern from "sumo/threedee/point_cloud_ply_c.h" namespace "sumo":

  # This is a wrapper for the function readPoints
  cdef vector[float] readPoints(const char* filename)

  # This is a wrapper for the function writePoints
  cdef void writePoints(
    vector[float]& vertex_points,
    const char* filename)

  # This is a wrapper for the function writePointsAndColor
  cdef void writePointsAndColors(
    vector[float]& vertex_points,
    vector[unsigned char]& vertex_colors,
    const char* filename)
