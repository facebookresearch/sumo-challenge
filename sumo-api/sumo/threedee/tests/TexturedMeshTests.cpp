/*
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include "sumo/threedee/TexturedMesh.h"

#include <gtest/gtest.h>

using namespace sumo;
using namespace std;

TEST(TexturedMesh, Constructor) {
  vector<Mesh::Index> indices(6);
  vector<Mesh::Vertex> vertices(4);
  vector<Mesh::Normal> normals(4);
  vector<TexturedMesh::UVCoord> uvCoords(4);
  cv::Size size(4, 3);
  cv::Mat3b baseColorTexture(size), metallicRoughnessTexture(size);
  TexturedMesh mesh(
      indices,
      vertices,
      normals,
      uvCoords,
      baseColorTexture,
      metallicRoughnessTexture);
  ASSERT_EQ(mesh.numIndices(), 6);
  ASSERT_EQ(mesh.numVertices(), 4);
  ASSERT_EQ(mesh.normals().size(), 4);
  ASSERT_EQ(mesh.uvCoords().size(), 4);
  ASSERT_EQ(mesh.baseColorTexture().size(), size);
  ASSERT_EQ(mesh.metallicRoughnessTexture().size(), size);
}

TEST(TexturedMesh, TestMergingSixMeshes) {
  vector<Mesh::Index> indices(6);
  vector<Mesh::Vertex> vertices(4);
  vector<Mesh::Normal> normals(4);
  vector<TexturedMesh::UVCoord> uvCoords(4);
  cv::Size size(4, 3);
  cv::Mat3b baseColorTexture(size), metallicRoughnessTexture(size);
  // Create 6 separate meshes
  vector<TexturedMesh> faces;
  for (size_t i = 0; i < 6; i++) {
    faces.emplace_back(
        indices,
        vertices,
        normals,
        uvCoords,
        baseColorTexture,
        metallicRoughnessTexture);
  }
  // merge faces:
  TexturedMesh mesh = faces[5]; // copy constructor
  for (size_t i = 0; i < 5; i++) {
    mesh.merge(faces[i], 0);
  }
  ASSERT_EQ(mesh.numIndices(), 36);
  ASSERT_EQ(mesh.numVertices(), 24);
  ASSERT_EQ(mesh.normals().size(), 24);
  ASSERT_EQ(mesh.uvCoords().size(), 24);
  ASSERT_EQ(mesh.baseColorTexture().size(), size);
  ASSERT_EQ(mesh.metallicRoughnessTexture().size(), size);
}
