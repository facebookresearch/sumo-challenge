/*
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>

#include "GltfModel.h"
#include "TexturedMesh.h"

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE // We are writing our own images
#include "sumo/threedee/tiny_gltf/tiny_gltf.h"

#define STB_IMAGE_IMPLEMENTATION
#include "sumo/threedee/tiny_gltf/stb_image.h"

namespace sumo {

using namespace std;

// Small templated funbction that returns type constant for given type.
template <class T>
static int ComponentType() {
  throw runtime_error("ComponentType::unhandled type");
}

template <>
int ComponentType<unsigned int>() {
  return TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
}

template <>
int ComponentType<float>() {
  return TINYGLTF_COMPONENT_TYPE_FLOAT;
}

// Adds raw data to buffer[0]
size_t GltfModel::reallyAddToBuffer(
    const unsigned char* rawData,
    size_t numBytes) {
  assert(!this->buffers.empty());
  vector<unsigned char>& bufferData = this->buffers[0].data;

  // Push data on buffer
  size_t byteOffset = bufferData.size();
  bufferData.resize(byteOffset + numBytes);
  memcpy(bufferData.data() + byteOffset, rawData, numBytes);
  return byteOffset;
}

// Adds data to buffer[0]
// Default template, assumes T is scalar
template <class T>
pair<size_t, size_t> GltfModel::addDataToBuffer(const vector<T>& values) {
  size_t count = values.size();
  size_t numBytes = count * sizeof(T);
  size_t byteOffset = reallyAddToBuffer(
      reinterpret_cast<const unsigned char*>(values.data()), numBytes);
  return {byteOffset, numBytes};
}

// specialization to Vector3f
template <>
pair<size_t, size_t> GltfModel::addDataToBuffer(
    const vector<Eigen::Vector3f>& values) {
  size_t count = values.size();
  size_t numBytes = count * sizeof(float) * 3;
  size_t byteOffset = reallyAddToBuffer(
      reinterpret_cast<const unsigned char*>(values[0].data()), numBytes);
  return {byteOffset, numBytes};
}

// Complete accessor with type-specific fields, scalar version
template <class T>
static void completeAccessor(
    const vector<T>& values,
    tinygltf::Accessor* accessor) {
  accessor->componentType = ComponentType<T>();
  accessor->type = TINYGLTF_TYPE_SCALAR;
  T minValue = (T)INT_MAX, maxValue = 0;
  for (size_t i = 0; i < accessor->count; i++) {
    minValue = min(minValue, values[i]);
    maxValue = max(maxValue, values[i]);
  }
  accessor->minValues.push_back(static_cast<double>(minValue));
  accessor->maxValues.push_back(static_cast<double>(maxValue));
}

// Complete accessor with type-specific fields, Vector3f version
template <>
void completeAccessor(
    const vector<Vector3f>& values,
    tinygltf::Accessor* accessor) {
  accessor->componentType = ComponentType<float>();
  accessor->type = TINYGLTF_TYPE_VEC3;
  for (size_t c = 0; c < 3; c++) {
    float minValue = INT_MAX, maxValue = INT_MIN;
    for (size_t i = 0; i < accessor->count; i++) {
      minValue = min(minValue, values[i][c]);
      maxValue = max(maxValue, values[i][c]);
    }
    accessor->minValues.push_back(static_cast<double>(minValue));
    accessor->maxValues.push_back(static_cast<double>(maxValue));
  }
}

// Complete accessor with type-specific fields, Vector2f version
template <>
void completeAccessor(
    const vector<Vector2f>& values,
    tinygltf::Accessor* accessor) {
  accessor->componentType = ComponentType<float>();
  accessor->type = TINYGLTF_TYPE_VEC2;
  for (size_t c = 0; c < 2; c++) {
    float minValue = INT_MAX, maxValue = INT_MIN;
    for (size_t i = 0; i < accessor->count; i++) {
      minValue = min(minValue, values[i][c]);
      maxValue = max(maxValue, values[i][c]);
    }
    accessor->minValues.push_back(static_cast<double>(minValue));
    accessor->maxValues.push_back(static_cast<double>(maxValue));
  }
}

template <class T>
GltfAccessor<T>::GltfAccessor(
    const GltfModel& object,
    const vector<T>& values,
    int bufferViewIndex) {
  tinyAccessor_.count = values.size();
  tinyAccessor_.byteOffset = 0;
  tinyAccessor_.bufferView = bufferViewIndex;
  completeAccessor<T>(values, &tinyAccessor_);
  calculatePointer(object);
}

template <class T>
int GltfModel::addAccessor(const vector<T>& values, int bufferViewIndex) {
  int accessorIndex = this->accessors.size();
  GltfAccessor<T> accessor(*this, values, bufferViewIndex);
  this->accessors.push_back(accessor.tinyAccessor());
  return accessorIndex;
}

int GltfModel::reallyAddBufferView(
    size_t numBytes,
    int target, // TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER (for indices)|
                // TINYGLTF_TARGET_ARRAY_BUFFER (for vertex data)
    size_t byteOffset,
    size_t byteStride) {
  int bufferViewIndex = this->bufferViews.size();
  tinygltf::BufferView view;
  view.buffer = 0;
  view.byteOffset = byteOffset;
  view.byteLength = numBytes;
  view.byteStride = byteStride;
  view.target = target;
  this->bufferViews.push_back(view);
  return bufferViewIndex;
}

// Add a view on the first buffer, sacalar
template <class T>
int GltfModel::addBufferView(size_t numBytes, int target, size_t byteOffset) {
  return reallyAddBufferView(numBytes, target, byteOffset, 0);
}

// Add a view on the first buffer, Vector3f
template <>
int GltfModel::addBufferView<Vector3f>(
    size_t numBytes,
    int target,
    size_t byteOffset) {
  return reallyAddBufferView(numBytes, target, byteOffset, 3 * sizeof(float));
}

template <class T>
int GltfModel::pushData(int target, const vector<T>& values) {
  size_t byteOffset, numBytes;
  tie(byteOffset, numBytes) = addDataToBuffer<T>(values);
  int bufferViewIndex = addBufferView<T>(numBytes, target, byteOffset);
  return addAccessor<T>(values, bufferViewIndex);
}

int GltfModel::addTexture(const cv::Mat3b& cvImage) {
  // Add image
  tinygltf::Image image;
  image.uri = to_string(images.size()) + ".png"; // Matches saveImages
  image.width = cvImage.cols;
  image.height = cvImage.rows;
  image.component = 3;
  size_t size = static_cast<size_t>(image.width * image.height * 3);
  image.image.resize(size);
  copy(cvImage.data, cvImage.data + size, image.image.begin());
  this->images.push_back(image);

  // Add texture
  tinygltf::Texture texture;
  texture.source = this->images.size() - 1;
  texture.sampler = 0;
  this->textures.push_back(texture);

  // Return texture id
  return this->textures.size() - 1;
}

static cv::Mat3b cvImageFromImage(const tinygltf::Image& image) {
  // Check image depth
  if (image.component != 3 && image.component != 4) {
    throw runtime_error(
        "GltfModel::cvImageFromImage: unexpected image depth " +
        to_string(image.component) + " for image with uri '" + image.uri +
        "'.");
  }

  // Check image dimensions
  size_t size = static_cast<size_t>(image.width * image.height);
  if (image.image.size() != size * image.component) {
    cerr << "  found image size " << image.image.size() << endl;
    cerr << "  expected image size " << size * image.component << endl;
    throw runtime_error(
        "GltfModel::cvImageFromImage: unexpected image dimensions"
        " for image with uri '" +
        image.uri + "'.");
  }

  // Create OpenCV image. TODO: discards alpha channel for now.
  cv::Mat3b cvImage(image.height, image.width);
  if (image.component == 3) {
    copy(image.image.begin(), image.image.end(), cvImage.data);
  } else {
    cv::Mat4b srcRgba(image.height, image.width);
    copy(image.image.begin(), image.image.end(), srcRgba.data);
    cv::cvtColor(srcRgba, cvImage, CV_RGBA2RGB);
  }
  return cvImage;
}

cv::Mat3b GltfModel::extractTexture(int textureId) const {
  // Get texture
  if (textureId < 0 || textureId >= static_cast<int>(this->textures.size())) {
    throw invalid_argument("GltfModel::extractTexture: invalid textureId.");
  }
  const tinygltf::Texture& texture = this->textures[textureId];

  // Get image
  if (texture.source >= static_cast<int>(this->images.size())) {
    throw runtime_error("GltfModel::extractTexture: invalid texture found.");
  }
  const tinygltf::Image& image = this->images[texture.source];

  return cvImageFromImage(image);
}

void GltfModel::addPrimitiveMesh(const Mesh& mesh) {
  //Add Mesh to model.
  // Create buffer.
  if (this->buffers.size() == 0) {
    this->buffers.emplace_back();
  }
  if (this->scenes.size() == 0) {
    this->scenes.emplace_back();
  }
  this->defaultScene = 0;

  // Add data to buffer, with buffer views and accessors.
  int indicesAccessorIndex = pushData<unsigned int>(
      TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER, mesh.indices());
  int vertexAccessorIndex =
      pushData<Vector3f>(TINYGLTF_TARGET_ARRAY_BUFFER, mesh.vertices());
  int normalAccessorIndex =
      pushData<Vector3f>(TINYGLTF_TARGET_ARRAY_BUFFER, mesh.normals());

  // Add index of node to scene.
  this->scenes[0].nodes.push_back(this->nodes.size());

  // Create new node and add index of mesh to be created.
  tinygltf::Node node;
  node.mesh = this->meshes.size();
  this->nodes.push_back(node);

  // Create mesh primitive.
  tinygltf::Primitive primitive;
  primitive.attributes["POSITION"] = vertexAccessorIndex;
  primitive.attributes["NORMAL"] = normalAccessorIndex;
  primitive.indices = indicesAccessorIndex;
  primitive.material = this->materials.size();
  primitive.mode = TINYGLTF_MODE_TRIANGLES;

  // Create mesh.
  tinygltf::Mesh tmesh;
  tmesh.primitives.push_back(primitive);
  this->meshes.push_back(tmesh);

  // Material for textured mesh.
  tinygltf::Material material = mesh.material_;
  this->materials.push_back(material);
}

void GltfModel::addTexturedPrimitiveMesh(const TexturedMesh& mesh) {
  //Add TexturedMesh to model.
  addPrimitiveMesh(mesh);

  // uv coordinates to primitives
  int uvCoordAccessorIndex =
      pushData<Vector2f>(TINYGLTF_TARGET_ARRAY_BUFFER, mesh.uvCoords());

  tinygltf::Mesh tmesh = this->meshes.back();
  tinygltf::Primitive primitive = tmesh.primitives[0];
  primitive.attributes["TEXCOORD_0"] = uvCoordAccessorIndex;
  tmesh.primitives.pop_back();
  tmesh.primitives.push_back(primitive);
  this->meshes.pop_back();
  this->meshes.push_back(tmesh);

  // Material for textured mesh.
  tinygltf::Material material = mesh.material_;

  // Add color texture if it exists in mesh
  if (!mesh.baseColorTexture().empty()) {
    if (material.values.count("baseColorTexture")) {
      tinygltf::Parameter color;
      color.json_double_value["index"] = addTexture(mesh.baseColorTexture_);
      material.values["baseColorTexture"] = color;
    } else {
      throw runtime_error("addTexturedPrimitiveMesh: inconsistent material.");
    }
  }

  // Add metallic/roughness if it exists in mesh
  if (!mesh.metallicRoughnessTexture().empty()) {
    if (material.values.count("metallicRoughnessTexture")) {
      tinygltf::Parameter roughness;
      roughness.json_double_value["index"] =
          addTexture(mesh.metallicRoughnessTexture_);
      material.values["metallicRoughnessTexture"] = roughness;
    } else {
      throw runtime_error("addTexturedPrimitiveMesh: inconsistent material.");
    }
  }
  this->materials.pop_back();
  this->materials.push_back(material);

  // Add default sampler
  //TODO: Read sampler from input glTF file and write back the same sampler.
  if (this->samplers.size() == 0) {
    tinygltf::Sampler sampler;
    sampler.minFilter = TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR;
    sampler.magFilter = TINYGLTF_TEXTURE_FILTER_LINEAR;
    sampler.wrapS = TINYGLTF_TEXTURE_WRAP_REPEAT;
    sampler.wrapT = TINYGLTF_TEXTURE_WRAP_REPEAT;
    this->samplers.push_back(sampler);
  }

}

size_t GltfModel::numPrimitiveMeshes() const {
  size_t index = 0;
  for (size_t i = 0; i < this->meshes.size(); i++) {
    const tinygltf::Mesh& mesh = this->meshes[i];
    for (size_t j = 0; j < mesh.primitives.size(); j++) {
      index++;
    }
  }
  return index;
}

const tinygltf::Primitive& GltfModel::getPrimitiveMesh(
    const size_t meshIndex) const {
  size_t index = 0;
  for (size_t i = 0; i < this->meshes.size(); i++) {
    const tinygltf::Mesh& mesh = this->meshes[i];
    for (size_t j = 0; j < mesh.primitives.size(); j++) {
      if (index == meshIndex) {
        return mesh.primitives[j];
      }
      index++;
    }
  }
  throw runtime_error("GltfModel::getPrimitiveMesh: meshIndex out of bounds.");
}

void GltfModel::extractPrimitiveMesh(const size_t meshIndex, Mesh* mesh) const {
  const tinygltf::Primitive& primitive = this->getPrimitiveMesh(meshIndex);

  // Get indices
  int indicesAccessorIndex = primitive.indices;
  int component_type = getAccessorType(*this, indicesAccessorIndex);
  // If component type is 16-bit, convert to 32 bit since Mesh indices are
  // 32 bit.
  if (component_type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
    auto indexAccessor =
        getAccessor<unsigned short>(*this, indicesAccessorIndex);
    vector<unsigned short> short_indices = indexAccessor.asVector();
    vector<Mesh::Index> mesh_indices(
        short_indices.begin(), short_indices.end());
    mesh->indices_ = mesh_indices;
  } else {
    auto indexAccessor = getAccessor<unsigned int>(*this, indicesAccessorIndex);
    mesh->indices_ = indexAccessor.asVector();
  }
  // Get vertices
  int vertexAccessorIndex = primitive.attributes.at("POSITION");
  auto vertexAccessor = getAccessor<Mesh::Vertex>(*this, vertexAccessorIndex);
  mesh->vertices_ = vertexAccessor.asVector();
  // Get normals
  int normalAccessorIndex = primitive.attributes.at("NORMAL");
  auto normalAccessor = getAccessor<Mesh::Vertex>(*this, normalAccessorIndex);
  mesh->normals_ = normalAccessor.asVector();

  // Get material
  int materialId = primitive.material;
  if (materialId < 0 ||
      materialId >= static_cast<int>(this->materials.size())) {
    throw runtime_error(
        "GltfModel::extractTexturedMesh: invalid material index.");
  }
  const tinygltf::Material& material = this->materials[materialId];

  // Copy material to mesh
  mesh->material_ = material;
}

void GltfModel::extractTexturedPrimitiveMesh(
    const size_t meshIndex,
    TexturedMesh* mesh) const {
  extractPrimitiveMesh(meshIndex, mesh); // Extract indices, vertices, normals.

  const tinygltf::Primitive& primitive = this->getPrimitiveMesh(meshIndex);

  // Get uv coordinates
  if (primitive.attributes.find("TEXCOORD_0") == primitive.attributes.end()) {
    throw runtime_error(
        "GltfModel::extractTexturedMesh: Texture coordinates not found.");
  }
  int uvCoordAccessorIndex = primitive.attributes.at("TEXCOORD_0");
  auto uvCoordAccessor =
      getAccessor<TexturedMesh::UVCoord>(*this, uvCoordAccessorIndex);
  mesh->uvCoords_ = uvCoordAccessor.asVector();

  // Get material
  int materialId = primitive.material;
  if (materialId < 0 ||
      materialId >= static_cast<int>(this->materials.size())) {
    throw runtime_error(
        "GltfModel::extractTexturedMesh: invalid material index.");
  }
  const tinygltf::Material& material = this->materials[materialId];

  // Extract textures if they exist, otherwise empty
  auto it_color = material.values.find("baseColorTexture");
  if (it_color != material.values.end()) {
    const tinygltf::Parameter& color = it_color->second;
    mesh->baseColorTexture_ =
        extractTexture(color.json_double_value.at("index"));
  }

  // Metallic-roughness texture
  auto it = material.values.find("metallicRoughnessTexture");
  if (it != material.values.end()) {
    const tinygltf::Parameter& roughness = it->second;
    mesh->metallicRoughnessTexture_ =
        extractTexture(roughness.json_double_value.at("index"));
  }
}

vector<Mesh*> GltfModel::getPolymorphicPrimitiveMeshes() const {
  vector<Mesh*> meshes;
  for (size_t i = 0; i < this->numPrimitiveMeshes(); i++) {
    const tinygltf::Primitive& primitive = this->getPrimitiveMesh(i);
    // Distinguish textured mesh or mesh by whether it has texture coords
    if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
      auto mesh = new TexturedMesh();
      extractTexturedPrimitiveMesh(i, mesh);
      meshes.push_back(mesh);
    } else {
      auto mesh = new Mesh();
      extractPrimitiveMesh(i, mesh);
      meshes.push_back(mesh);
    }
  }
  return meshes;
}

void GltfModel::setURIs(const string& extension) {
  for (size_t i = 0; i < this->images.size(); i++) {
    auto& image = this->images[i];
    if (!image.uri.empty()) {
      throw invalid_argument(
          "GltfModel::setURIs: expected empty URI, but got ''" + image.uri +
          "'.");
    }
    image.uri = to_string(i) + extension;
  }
}

void GltfModel::saveImages(const string& folder) const {
  for (size_t i = 0; i < this->images.size(); i++) {
    const auto& image = this->images[i];
    if (image.image.data() == nullptr) {
      throw runtime_error(
          "GltfModel::saveImages: has no images loaded in memory.");
    }
    cv::Mat3b cvImage = cvImageFromImage(image);
    cv::Mat3b bgr(image.height, image.width);
    cv::cvtColor(cvImage, bgr, cv::COLOR_RGB2BGR);
    if (image.uri.empty()) {
      throw invalid_argument("GltfModel::saveImages: found empty image URI");
    }
    try {
      cv::imwrite(folder + "/" + image.uri, bgr);
    } catch (cv::Exception& e) {
      throw runtime_error(
          "GltfModel::saveImages: OpenCV could not write " + image.uri +
          ", reason: " + e.what());
    }
  }
}

void GltfModel::updateMaterial(
  size_t primitiveMeshIndex,
  const Vector3d& color,
  const string& uri,
  const string& baseDir) {

  if (primitiveMeshIndex >= this->numPrimitiveMeshes()) {
    throw invalid_argument(
        "GltfModel::updateMaterial: invalid meshIndex: " +
        to_string(primitiveMeshIndex));
  }

  // Run through and find the primitive mesh and the material
  tinygltf::Material material;
  size_t meshIndex = 0;
  size_t primitiveIndex = 0;
  size_t index = 0;
  for (size_t i = 0; i < this->meshes.size(); i++) {
    const tinygltf::Mesh& mesh = this->meshes[i];
    for (size_t j = 0; j < mesh.primitives.size(); j++) {
      if (index == primitiveMeshIndex) {
        size_t materialIndex = mesh.primitives[j].material;
        if (materialIndex >= this->materials.size()) {
          throw runtime_error(
              "GltfModel::updateMaterial: material index out of bounds " +
              to_string(materialIndex));
        }
        // Copy the material
        material = this->materials[materialIndex];
        meshIndex = i;
        primitiveIndex = j;
      }
      index++;
    }
  }

  // color == [0, 0, 0] implies no color, uri.size() == 0,
  // implies no texture
  if (color[0] != 0 && color[1] != 0 && color[2] != 0) {
    // Update the color
    tinygltf::Parameter baseColorFactor;
    baseColorFactor.number_array = {color[0], color[1], color[2], 1};
    material.values["baseColorFactor"] = baseColorFactor;
  }
  if (uri.size() != 0) {
    // Update the texture
    tinygltf::Parameter baseColorTexture;
    size_t textureIndex = this->loadTexture(uri, baseDir);
    baseColorTexture.json_double_value["index"] = textureIndex;
    material.values["baseColorTexture"] = baseColorTexture;
  }

  // Add the material to the set of materials, and update the primitive
  // Follow the naming convention in our objects for material names
  material.name =
      string("material_") + to_string(this->materials.size());
  this->materials.push_back(material);
  tinygltf::Mesh& mesh = this->meshes[meshIndex];
  tinygltf::Primitive& primitive = mesh.primitives[primitiveIndex];
  primitive.material = this->materials.size() - 1;
}

size_t GltfModel::loadTexture(
    const string& uri,
    const string& baseDir) {

  string filename = baseDir + "/" + uri;

  vector<unsigned char> bytes;
  if (!tinygltf::ReadWholeFile(&bytes, nullptr, filename, nullptr)) {
    throw runtime_error("loadTexture: read file: " + filename + " failed");
  }
  if (bytes.empty()) {
    throw runtime_error("loadTexture: Empty file: " + filename);
  }
  tinygltf::Image image;
  tinygltf::LoadImageData(
      &image,
      nullptr,
      0,
      0,
      &bytes.at(0),
      static_cast<int>(bytes.size()),
      nullptr);

  size_t start = uri.find_last_of("/") + 1;
  size_t end = uri.find_last_of(".");
  // This should always be true, but just being paranoid
  if (start != string::npos && end != string::npos) {
    image.name = uri.substr(start, end-start);
  }
  image.uri = to_string(images.size()) + ".png"; // Matches saveImages
  this->images.push_back(image);

  // Add texture
  tinygltf::Texture texture;
  texture.source = this->images.size() - 1;
  texture.sampler = 0;  // Default sampler.
  this->textures.push_back(texture);

  // Return texture id
  return this->textures.size() - 1;
}

size_t GltfModel::addColoredMaterial(
    const string& name,
    const Vector3d& color,
    double metallic,
    double roughness) {
  tinygltf::Parameter baseColorFactor;
  tinygltf::Parameter metallicFactor, roughNessFactor;
  baseColorFactor.number_array = {color[0], color[1], color[2], 1};
  metallicFactor.number_value = metallic;
  roughNessFactor.number_value = roughness;
  tinygltf::Material material;
  material.name = name;
  material.values["baseColorFactor"] = baseColorFactor;
  material.values["metallicFactor"] = metallicFactor;
  material.values["roughNessFactor"] = roughNessFactor;

  this->materials.push_back(material);
  // Return material id
  return this->materials.size() - 1;
}

int getAccessorType(const GltfModel& object, size_t i) {
  if (i >= object.accessors.size()) {
    throw std::invalid_argument("GltfAccessor: accessor index out of bounds");
  }
  return object.accessors[i].componentType;
}

} // namespace sumo
