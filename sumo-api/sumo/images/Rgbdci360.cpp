/*
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include "sumo/base/Vector.h"
#include "sumo/images/CubeMap.h"
#include "sumo/images/Rgbdci360.h"
#include "sumo/threedee/PointCloud.h"

#include <vector>

using namespace std;

namespace sumo {

Vector3 pointFromRange(
    const CubeMap& cube_map,
    const Vector2& pixel,
    double range)
{
  Vector3 ray = cube_map.unproject(pixel);
  return ray * (range / ray.norm());
}

PointCloud* createPointCloud(
    const cv::Mat3b& rgb,
    const cv::Mat1f& range,
    bool all_points)
{
  size_t width = rgb.cols, height = rgb.rows;
  vector<Vector3> points;
  points.reserve(height * width);
  vector<PointCloud::Color> colors;
  colors.reserve(height * width);
  // ::: can rgb and range be different?
  CubeMap cube_map = CubeMap(height);

  for (size_t row = 0; row < height; ++row) {
    Vector2 pixel = CubeMap::pixelCoordinates(row, 0);
    for (size_t col = 0; col < width; ++col) {
      float range_value = range(row, col);
      if (range_value > 0.0 and range_value != INFINITY) {
        Vector3 point = pointFromRange(cube_map, pixel, range_value);
        points.push_back(point);
        auto color = rgb(row, col);
        colors.emplace_back(color[0], color[1], color[2]);
      } else if (all_points) {
        points.push_back(Vector3(0, 0, 0));
        colors.emplace_back(0, 0, 0);
      }
      pixel[0] += 1.0;
    }
  }

  return new PointCloud(points, colors);
}

} // namespace sumo
