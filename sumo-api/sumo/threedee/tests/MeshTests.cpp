/*
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

// Without this flag, GTSAM Expressions currently fail within fbcode.
// We need EIGEN_MAKE_ALIGNED_OPERATOR_NEW in GTSAM to really fix this
#define EIGEN_DONT_ALIGN_STATICALLY

#include <sumo/threedee/Mesh.h>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

using namespace sumo;
using namespace std;

const double kSideLength = 3.0;

// Test fixture class
class MeshTest : public ::testing::Test {
 protected:
  Mesh cubeMesh_;

  MeshTest() {}

  ~MeshTest() {}

  virtual void SetUp() override {
    cubeMesh_ = Mesh::Example(kSideLength, true);
  }
};

TEST_F(MeshTest, ColoredCubeSanity) {
  ASSERT_EQ(cubeMesh_.numIndices(), 12 * 3);
  ASSERT_EQ(cubeMesh_.numVertices(), 8);
  ASSERT_EQ(cubeMesh_.normals().size(), 8);
}

TEST_F(MeshTest, DefaultConstructor) {
  Mesh mesh = Mesh();
  ASSERT_EQ(mesh.indices().size(), 0);
  ASSERT_EQ(mesh.vertices().size(), 0);
  ASSERT_EQ(mesh.normals().size(), 0);
}

TEST_F(MeshTest, TriangularMeshIterator) {
  size_t index = 0;
  const auto& indices = cubeMesh_.indices();
  const auto& vertices = cubeMesh_.vertices();
  for (const auto& triangle : cubeMesh_) {
    ASSERT_EQ(triangle.i, indices[index]);
    ASSERT_EQ(triangle.j, indices[index + 1]);
    ASSERT_EQ(triangle.k, indices[index + 2]);
    ASSERT_TRUE(triangle.a == vertices[triangle.i]);
    ASSERT_TRUE(triangle.b == vertices[triangle.j]);
    ASSERT_TRUE(triangle.c == vertices[triangle.k]);
    index += 3;
  }
}
