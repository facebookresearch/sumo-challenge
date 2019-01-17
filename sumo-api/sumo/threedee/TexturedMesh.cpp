/*
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include "TexturedMesh.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace sumo {

using namespace std;

TexturedMesh::TexturedMesh(
    const vector<Index>& indices,
    const vector<Vertex>& vertices,
    const vector<Normal>& normals,
    const vector<UVCoord>& uvCoords,
    const cv::Mat3b& baseColorTexture,
    const cv::Mat3b& metallicRoughnessTexture)
    : Mesh(indices, vertices, normals),
      uvCoords_(uvCoords),
      baseColorTexture_(baseColorTexture.clone()),
      metallicRoughnessTexture_(metallicRoughnessTexture.clone()) {
  // Not all vertices necessarily have uv_coords
  assert(vertices.size() >= uvCoords.size());

  // Add color texture
  tinygltf::Parameter color;
  color.json_double_value["index"] = 0;
  material_.values["baseColorTexture"] = color;

  // Add metallic/roughness if it exists in mesh
  if (!metallicRoughnessTexture.empty()) {
    tinygltf::Parameter roughness;
    roughness.json_double_value["index"] = 1;
    material_.values["metallicRoughnessTexture"] = roughness;
  }
}

void TexturedMesh::renumber(
    size_t numNewVertices,
    vector<Index> vertexRenumbering) {
  // renumber indices
  vector<Index> newIndices;
  newIndices.reserve(indices_.size());
  const size_t numRenumbered = vertexRenumbering.size();
  for (const Index index : indices_) {
    newIndices.push_back(
        index < numRenumbered ? vertexRenumbering[index]
                              : index + numNewVertices);
  }
  indices_ = newIndices;

  // Assemble new vertex properties
  // Note: Have to explicitly list captures here due to (presumed) compiler
  // error that gives error (numRenumbered is not captured)
  const auto assemble = [numRenumbered, vertexRenumbering, numNewVertices]
    (auto& a, auto& b) {
    a.resize(numNewVertices);
    for (size_t i = 0; i < numRenumbered; i++) {
      a[vertexRenumbering[i]] = b[i];
    }
    a.insert(a.end(), b.begin() + numRenumbered, b.end());
    b = a;
  };
  vector<Vertex> newVertices;
  assemble(newVertices, vertices_);
  vector<Normal> newNormals;
  assemble(newNormals, normals_);
  vector<UVCoord> newUvCoords;
  assemble(newUvCoords, uvCoords_);
}

void TexturedMesh::merge(const TexturedMesh& mesh2, size_t numCommonVertices) {
  // Call base class method
  Mesh::merge(mesh2, numCommonVertices);

  // Treat the case when mesh2 defines new vertices.
  if (mesh2.numVertices() > numCommonVertices) {
    const auto append = [numCommonVertices](auto& a, const auto& b) {
      a.insert(a.end(), b.begin() + numCommonVertices, b.end());
    };
    append(uvCoords_, mesh2.uvCoords());
  }
}

void TexturedMesh::replaceGeometry(const TexturedMesh& mesh2) {
  indices_.clear();
  vertices_.clear();
  normals_.clear();
  uvCoords_.clear();
  merge(mesh2, 0);
}

bool TexturedMesh::hasDualTextureMaterial() const {
  auto it_color = material_.values.find("baseColorTexture");
  if (it_color == material_.values.end() ||
      it_color->second.json_double_value.at("index") != 0) {
    return false;
  }
  auto it = material_.values.find("metallicRoughnessTexture");
  if (it == material_.values.end() ||
      it->second.json_double_value.at("index") != 1) {
    return false;
  }
  return true;
}

} // namespace sumo
