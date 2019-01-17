/*
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#pragma once

#include <opencv2/core/core.hpp>

namespace sumo {

class PointCloud;

/**
 * Create point cloud from RGBD (where D is range image).
 *      rgb -- 3-channel uint8 RGB image
 *      range -- 1-channel float32 range image
 * Returns newly allocated PointCloud instance.
 */
PointCloud* createPointCloud(const cv::Mat3b& rgb, const cv::Mat1f& range,
  bool all_points);



} // sumo
