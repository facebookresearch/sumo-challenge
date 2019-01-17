#pragma once
/*
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include <string>
#include <vector>

namespace sumo {

/**
 * Read points from a PLY file.
 * Inputs:
 *  filename: file name of the PLY file
 * Output:
 *  unraveled points
 */
std::vector<float> readPoints(const std::string& filename);

/**
 * Write points to a PLY file.
 * Inputs:
 *  vertex_points: unraveled points
 *  filename: file name of the PLY file
 */
void writePoints(
    std::vector<float>& vertex_points,
    const std::string& filename);

/**
 * Write points and their colors to a PLY file.
 * Inputs:
 *  vertex_points: unraveled points
 *  vertex_colors: unraveled colors of the unraveled points
 *  filename: file name of the PLY file
 */
void writePointsAndColors(
    std::vector<float>& vertex_points,
    std::vector<unsigned char>& vertex_colors,
    const std::string& filename);

} // namespace sumo
