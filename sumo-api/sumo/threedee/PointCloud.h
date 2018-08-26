#pragma once

#include <vector>
#include "sumo/base/Vector.h"

namespace sumo {

class PointCloud {
 public:
  struct Color {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    Color() : r(0), g(0), b(0) {}
    Color(unsigned char r, unsigned char g, unsigned char b)
        : r(r), g(g), b(b) {}
    bool operator==(const Color& other) const {
      return r == other.r && g == other.g && b == other.b;
    }
  };

 private:
  std::vector<sumo::Vector3> points_;
  std::vector<Color> colors_;

 public:
  /// Nullary constructor for use in cython
  explicit PointCloud() {}

  /// Construct from points and optional colors
  explicit PointCloud(
      const std::vector<sumo::Vector3>& points,
      const std::vector<Color>& colors = {})
      : points_(points), colors_(colors) {
    if (!colors.empty()) {
      assert(points.size() == colors.size());
    }
  }

  /// Construct from a set of PointCloud instances. Pointers because cython.
  explicit PointCloud(const std::vector<const PointCloud*>& clouds) {
    size_t num_points = 0, num_colors = 0;
    for (const PointCloud* cloud : clouds) {
      num_points += cloud->points_.size();
      num_colors += cloud->colors_.size();
    }
    assert(num_colors == 0 || num_colors == num_points);
    points_.reserve(num_points);
    colors_.reserve(num_colors);
    for (const PointCloud* cloud : clouds) {
      points_.insert(
          points_.end(), cloud->points_.begin(), cloud->points_.end());
      colors_.insert(
          colors_.end(), cloud->colors_.begin(), cloud->colors_.end());
    }
  }

  // Append the points (and colors) of the other instance. Imperative.
  void append(const PointCloud& other) {
    points_.reserve(points_.size() + other.points_.size());
    points_.insert(points_.end(), other.points_.begin(), other.points_.end());
    colors_.reserve(colors_.size() + other.colors_.size());
    colors_.insert(colors_.end(), other.colors_.begin(), other.colors_.end());
  }

  const std::vector<sumo::Vector3>& points() const {
    return points_;
  }

  const sumo::Vector3& point(size_t i) const {
    return points_[i];
  }

  const std::vector<Color>& colors() const {
    return colors_;
  }

  const Color& color(size_t i) const {
    return colors_[i];
  }

  size_t numPoints() const {
    return points_.size();
  }

  bool colored() const {
    return !colors_.empty();
  }
};

} // namespace sumo
