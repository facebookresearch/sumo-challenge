/*
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include "point_cloud_ply_c.h"
#include "tinyply/tinyply.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

using namespace std;
using namespace tinyply;

namespace sumo {

// return a set of points that is read from a ply file specified by the filename
vector<float> readPoints(const string& filename) {
  // Tinyply can and will throw exceptions at you!

  // Read the file and create a istringstream suitable
  // for the lib -- tinyply does not perform any file i/o.
  ifstream ss(filename, ios::binary);

  if (!ss.good()) {
    throw runtime_error("readPoints error while reading " + filename);
  }

  // Parse the ASCII header fields
  PlyFile file;
  file.parse_header(ss);
  // The count returns the number of instances of the property group. The
  // vectors above will be resized into a multiple of the property group size
  // as they are "flattened"... i.e. vertices = {x, y, z, x, y, z, ...}
  shared_ptr<PlyData> vertexData =
      file.request_properties_from_element("vertex", {"x", "y", "z"});

  file.read(ss);

  const size_t numVerticesBytes = vertexData->buffer.size_bytes();
  vector<float> vertices(3 * vertexData->count);
  memcpy(vertices.data(), vertexData->buffer.get(), numVerticesBytes);

  return vertices;
}

void writePoints(vector<float>& vertex_points, const string& filename) {
  // Tinyply does not perform any file i/o internally
  filebuf fb;
  fb.open(filename, ios::out | ios::binary);

  if (!fb.is_open()) {
    throw runtime_error("writePoints error while opening " + filename);
  }

  PlyFile myFile;
  myFile.add_properties_to_element(
      "vertex",
      {"x", "y", "z"},
      Type::FLOAT32,
      vertex_points.size() / 3,
      reinterpret_cast<uint8_t*>(vertex_points.data()),
      Type::INVALID,
      0);
  ostream outputStream(&fb);
  myFile.write(outputStream, true);
  fb.close();
}

void writePointsAndColors(
    vector<float>& vertex_points,
    vector<unsigned char>& vertex_colors,
    const string& filename) {
  // Tinyply does not perform any file i/o internally
  filebuf fb;
  fb.open(filename, ios::out | ios::binary);

  if (!fb.is_open()) {
    throw runtime_error("writePoints error while opening " + filename);
  }

  PlyFile myFile;
  myFile.add_properties_to_element(
      "vertex",
      {"x", "y", "z"},
      Type::FLOAT32,
      vertex_points.size() / 3,
      reinterpret_cast<uint8_t*>(vertex_points.data()),
      Type::INVALID,
      0);
  myFile.add_properties_to_element(
      "vertex",
      {"red", "green", "blue"},
      Type::UINT8,
      vertex_colors.size() / 3,
      reinterpret_cast<uint8_t*>(vertex_colors.data()),
      Type::INVALID,
      0);
  ostream outputStream(&fb);
  myFile.write(outputStream, true);
  fb.close();
}

} // namespace sumo
