"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

"""

cimport numpy as np
from libcpp cimport bool

from sumo.threedee.point_cloud cimport CPointCloud
from sumo.opencv.wrap cimport Mat3b, Mat1f

cdef extern from "sumo/images/Rgbdci360.h" namespace "sumo":
  cdef CPointCloud* createPointCloud(const Mat3b& rgb,
                                     const Mat1f& range,
                                     bool all_points);
