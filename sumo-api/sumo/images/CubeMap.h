/*
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include "sumo/base/Vector.h"
#include <array>
#include <cmath>
#include <Eigen/Core>
#include <iostream>


namespace sumo {

typedef Eigen::Vector2d Vector2;
typedef Eigen::Vector3d Vector3;
typedef Eigen::Vector4d Vector4;

/// Camera model is classical pinhole model per face
class CubeMap {
 public:
  // Order of faces in the cube map
  enum Face { BACK, LEFT, FRONT, RIGHT, UP, DOWN };

  // side - side length of cube face (in pixels)
  explicit CubeMap(size_t side) :
    side_(side),
    ff_(Vector2(side / 2., side / 2.)),
    pp_(Vector2(side / 2., side / 2.))   {

    // Create horizontal layout
    for (size_t i = 0; i < 6; i++) {
      offset_[i] = Vector2(side * i, 0);
    }
  }

  /// Return size as (w,h) pair
  std::pair<size_t, size_t> size() const {
    return std::make_pair(side_ * 6, side_);
  }

  /* Implementation notes:

  The cube faces are defined by possibly half-open intervals, laid out as in the
  figure below, where '*' is closed, 'o' is open, and '+' is image origin.

            +   o
            o  U_P o
                o

            +   *
            * FRONT o
                *

      o     +   o         *   +
  * LEFT *  o DOWN o  * RIGHT *
  +   *         o         o

                *
            o BACK *
                *  +

  As you can see from the diagram above, we prioritize the vertical faces over
  the UP and DOWN faces, which have open boundaries on all sides.
  */


  /*
  The diagram above yields the following 6-face layout for unproject:
    +   *     +   *     +   *      +   *      +  x     +   x
    * BACK o  * LEFT o  * FRONT o  * RIGHT o  x U_P o  x DOWN x
        *         *         *          *         x         x
   Above, the 'x' edges are never projected to, but still need to be
   unprojected. We give the rightmost edge to the BACK face!
  */

  // Unproject from image to cube surface in 3D
  Vector3 unproject(const Vector2& pixel) const {
    if (pixel.x() == 6 * side_) {
      Vector2 sensor = sensorOfPixel(Vector2(0, pixel.y()));
      return {-sensor.x(), sensor.y(), -1};
    } else {
      size_t face = (size_t)(pixel.x() / side_);
      Vector2 sensor = sensorOfPixel(pixel - offset_.at(face));
      switch (face) {
        case BACK:
          return {-sensor.x(), sensor.y(), -1};
        case LEFT:
          return {-1, sensor.y(), sensor.x()};
        case FRONT:
          return {sensor.x(), sensor.y(), 1};
        case RIGHT:
          return {1, sensor.y(), -sensor.x()};
        case UP:
          return {sensor.x(), -1, sensor.y()};
        case DOWN:
          return {sensor.x(), 1, -sensor.y()};
        default:
          throw std::runtime_error("CubeMap::unproject pixel out of range");
      }
    }
  }

  Vector2 sensorOfPixel(const Vector2& pixel) const {
    return (pixel - pp_).cwiseQuotient(ff_);
  }


  /// Return floating point vector for middle of pixel with integer coordinates
  /// <row,col>
  static Vector2 pixelCoordinates(const size_t row, const size_t col)
  {
    return Vector2(static_cast<double>(col) + 0.5,
                   static_cast<double>(row) + 0.5);
  }

 private:
  // cube face = side * side
  size_t side_;

  Vector2 ff_; // focal lengths
  Vector2 pp_; // principal point

  // array of offsets to add to pixel values to create compound cubemap image
  std::array<Vector2, 6> offset_;
}; // CubeMap

} // namespace sumo
