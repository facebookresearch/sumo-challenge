// Copyright 2004-present Facebook. All Rights Reserved.

#include <gtest/gtest.h>

#include "sumo/threedee/GltfModel.h"
#include "sumo/threedee/TexturedMesh.h"

namespace sumo {

using namespace std;

TEST(TinyGltf, SimpleModel) {
  // Test that creating a simple GltfModel compiles and links.
  GltfModel model;
  ASSERT_EQ(model.nodes.size(), 0);
}

TEST(TinyGltf, reallyAddToBuffer) {
  GltfModel model;
  model.buffers.emplace_back();
  const std::vector<unsigned char> rawData = {'a', 'b', 'c'};
  size_t byteOffset = model.reallyAddToBuffer(rawData.data(), 3);
  ASSERT_EQ(byteOffset, 0);
  ASSERT_EQ(model.buffers.size(), 1);
  ASSERT_EQ(model.buffers[0].data.size(), 3);
  ASSERT_EQ(model.buffers[0].data[0], 'a');
}

TEST(TinyGltf, addDataToBuffer) {
  GltfModel model;
  model.buffers.emplace_back();
  const std::vector<unsigned int> typedData = {1, 2, 3};
  size_t byteOffset, numBytes;
  std::tie(byteOffset, numBytes) = model.addDataToBuffer(typedData);
  ASSERT_EQ(byteOffset, 0);
  ASSERT_EQ(numBytes, 12);
  ASSERT_EQ(model.buffers.size(), 1);
  const auto& data = model.buffers[0].data;
  ASSERT_EQ(data.size(), 12);
  // Note, Buffer data is little endian !!
  ASSERT_EQ(data[0], 1);
  ASSERT_EQ(data[1], 0);
  ASSERT_EQ(data[4], 2);
  ASSERT_EQ(data[5], 0);
  ASSERT_EQ(data[8], 3);
  ASSERT_EQ(data[9], 0);
  auto pointer = reinterpret_cast<const unsigned int*>(data.data());
  ASSERT_EQ(pointer[0], 1);
  ASSERT_EQ(pointer[1], 2);
  ASSERT_EQ(pointer[2], 3);
}

TEST(TinyGltf, pushAccessor) {}

TEST(TinyGltf, pushData) {
  GltfModel model;
  model.buffers.emplace_back();
  const std::vector<unsigned int> typedData = {1, 2, 3};
  int accessorIndex =
      model.pushData(TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER, typedData);

  ASSERT_EQ(accessorIndex, 0);

  ASSERT_EQ(model.buffers.size(), 1);
  const auto& data = model.buffers[0].data;
  ASSERT_EQ(data.size(), 12);

  // Note, Buffer data is little endian !!
  ASSERT_EQ(data[0], 1);
  ASSERT_EQ(data[1], 0);

  ASSERT_EQ(model.bufferViews.size(), 1);
  ASSERT_EQ(model.bufferViews[0].byteOffset, 0);
  ASSERT_EQ(model.bufferViews[0].byteLength, 12);

  ASSERT_EQ(model.accessors.size(), 1);
  ASSERT_EQ(model.accessors[0].bufferView, 0);

  // Test accessor on same data
  auto accessor = getAccessor<unsigned int>(model, accessorIndex);
  ASSERT_EQ(accessor[0], 1);
  ASSERT_EQ(accessor[1], 2);
  ASSERT_EQ(accessor[2], 3);
}

TEST(TinyGltf, pushDataTwice) {
  GltfModel model;
  model.buffers.emplace_back();

  const std::vector<unsigned int> indices = {1, 2, 3};
  int indicesAccessorIndex = model.pushData<unsigned int>(
      TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER, indices);
  ASSERT_EQ(indicesAccessorIndex, 0);

  vector<Mesh::Vertex> vertices(6); // dummy values
  vertices[0] = Mesh::Vertex(1.1f, 2.2f, 3.3f);
  int vertexAccessorIndex =
      model.pushData<Vector3f>(TINYGLTF_TARGET_ARRAY_BUFFER, vertices);
  ASSERT_EQ(vertexAccessorIndex, 1);

  ASSERT_EQ(model.buffers.size(), 1);
  const auto& data = model.buffers[0].data;
  ASSERT_EQ(data.size(), 3 * 4 + 6 * 3 * 4);

  auto indexAccessor = getAccessor<unsigned int>(model, indicesAccessorIndex);
  ASSERT_EQ(indexAccessor[0], 1);
  ASSERT_EQ(indexAccessor[1], 2);
  ASSERT_EQ(indexAccessor[2], 3);

  // TODO: why are these values not exact??
  auto coords = getAccessor<float>(model, vertexAccessorIndex);
  ASSERT_NEAR(coords[0], 1.1, 1e-7);
  ASSERT_NEAR(coords[1], 2.2, 1e-7);
  ASSERT_NEAR(coords[2], 3.3, 1e-7);
}

TEST(TinyGltf, addTexturedPrimitiveMesh) {
  // Check some aspects of adding mesh data to the GLTF model.
  GltfModel model;
  const vector<Mesh::Index> indices = {1, 2, 3, 0, 1, 3};
  vector<Mesh::Vertex> vertices(4); // dummy values
  vertices[0] = Mesh::Vertex(1.1f, 2.2f, 3.3f);
  vertices[1] = Mesh::Vertex(4.4f, 5.5f, 6.6f);
  const vector<Mesh::Normal> normals(4);
  const vector<Vector2f> uvCoords(4);
  const cv::Size size(4, 3);
  const cv::Mat3b baseColorTexture(size), metallicRoughnessTexture(size);
  const TexturedMesh texturedMesh(
      indices,
      vertices,
      normals,
      uvCoords,
      baseColorTexture,
      metallicRoughnessTexture);
  model.addTexturedPrimitiveMesh(texturedMesh);

  const size_t indicesNumBytes = 6 * 4, verticesNumBytes = 4 * 4 * 3,
               uvCoordsNumBytes = 4 * 4 * 2;
  ASSERT_EQ(model.buffers.size(), 1);

  ASSERT_EQ(model.bufferViews.size(), 4);
  ASSERT_EQ(model.bufferViews[0].byteOffset, 0);
  ASSERT_EQ(model.bufferViews[0].byteLength, indicesNumBytes);
  ASSERT_EQ(model.bufferViews[1].byteOffset, indicesNumBytes);
  ASSERT_EQ(model.bufferViews[1].byteLength, verticesNumBytes);
  ASSERT_EQ(
      model.bufferViews[2].byteOffset, indicesNumBytes + verticesNumBytes);
  ASSERT_EQ(model.bufferViews[2].byteLength, verticesNumBytes);
  ASSERT_EQ(
      model.bufferViews[3].byteOffset, indicesNumBytes + 2 * verticesNumBytes);
  ASSERT_EQ(model.bufferViews[3].byteLength, uvCoordsNumBytes);

  ASSERT_EQ(model.accessors.size(), 4);
  ASSERT_EQ(model.accessors[0].bufferView, 0);
  ASSERT_EQ(model.accessors[1].bufferView, 1);
  ASSERT_EQ(model.accessors[2].bufferView, 2);
  ASSERT_EQ(model.accessors[3].bufferView, 3);

  // test that the indices were correctly written
  auto indexAccessor = getAccessor<unsigned int>(model, 0);
  ASSERT_EQ(indexAccessor.size(), 6);
  ASSERT_EQ(indexAccessor[0], 1);
  ASSERT_EQ(indexAccessor[1], 2);
  ASSERT_EQ(indexAccessor[2], 3);

  // test that we can access vector-values types
  auto vertexAccessor = getAccessor<Mesh::Vertex>(model, 1);
  ASSERT_EQ(vertexAccessor.size(), 4);
  ASSERT_EQ(vertexAccessor[0], vertices[0]);
  ASSERT_EQ(vertexAccessor[1], vertices[1]);

  // test that we can get an entire array out of an accessor
  auto verticesFromAccessor = vertexAccessor.asVector();
  ASSERT_EQ(verticesFromAccessor, vertices);

  ASSERT_EQ(model.meshes.size(), 1);
  ASSERT_EQ(model.meshes[0].primitives.size(), 1);
  ASSERT_EQ(model.meshes[0].primitives[0].mode, 4);

  ASSERT_EQ(model.nodes.size(), 1);
  ASSERT_EQ(model.scenes.size(), 1);

  // Test extracting a mesh from the model. Assumes a single mesh.
  TexturedMesh mesh;
  model.extractTexturedPrimitiveMesh(0, &mesh);
  ASSERT_EQ(mesh.numIndices(), 6);
  ASSERT_EQ(mesh.indices(), indices);
  ASSERT_EQ(mesh.numVertices(), 4);
  ASSERT_EQ(mesh.vertices(), vertices);
  ASSERT_EQ(mesh.normals().size(), 4);
  ASSERT_EQ(mesh.normals(), normals);
  ASSERT_EQ(mesh.uvCoords().size(), 4);
  ASSERT_EQ(mesh.uvCoords(), uvCoords);
  ASSERT_EQ(mesh.baseColorTexture().size(), size);
  ASSERT_EQ(mesh.metallicRoughnessTexture().size(), size);
}

TEST(TinyGltf, addColorMaterial) {
  // Add basic color material to GltfModel
  GltfModel model;
  Vector3d color = {0.5, 0.5, 0.5};
  size_t index = model.addColoredMaterial(string("simple_color"), color);
  ASSERT_EQ(index, 0);
  ASSERT_EQ(model.materials.size(), 1);

}

TEST(TinyGltf, constructFromFile) {
  GltfModel modelBlind(std::string("sumo/threedee/test_data/blind.glb"));
  ASSERT_EQ(modelBlind.numPrimitiveMeshes(), 6);

  GltfModel modelCube(std::string("sumo/threedee/test_data/Cube.gltf"));
  ASSERT_EQ(modelCube.numPrimitiveMeshes(), 1);

}

} // namespace sumo
