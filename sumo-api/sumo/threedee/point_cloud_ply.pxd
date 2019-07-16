"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

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
