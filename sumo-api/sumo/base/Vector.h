#pragma once
/*
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

// Without this flag, GTSAM Expressions currently fail within fbcode.
// We need EIGEN_MAKE_ALIGNED_OPERATOR_NEW in GTSAM to really fix this
#define EIGEN_DONT_ALIGN_STATICALLY

#include <Eigen/Core>

namespace sumo {

typedef Eigen::Vector2d Vector2;
typedef Eigen::Vector2f Vector2f;
typedef Eigen::Vector3d Vector3;
typedef Eigen::Vector3f Vector3f;
typedef Eigen::Vector4d Vector4;
typedef Eigen::Matrix3d Matrix3;

} // namespace sumo
