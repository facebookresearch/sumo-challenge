/*
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#define TINYGLTF_NO_STB_IMAGE_WRITE // We are writing our own images

#include "sumo/threedee/tiny_gltf/stb_image.h"
#include "sumo/threedee/tiny_gltf/tiny_gltf.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Core>

#include <stdexcept>
#include <string>
#include <vector>

namespace sumo {

class Mesh;
class TexturedMesh;
typedef Eigen::Vector3d Vector3d;
typedef Eigen::Vector3f Vector3f;

/**
 * Our own more functional class on top of tinygltf::Model.
 * Main entry point is addMesh, other methods are advanced/internal.
 */
class GltfModel : public tinygltf::Model {
 public:
  /// Default constructor
  GltfModel() {}

  /**
   * Constructor from a glb or gltf file
   */
  explicit GltfModel(std::string filename) {
    std::string err;
    tinygltf::TinyGLTF tiny;
    // Use file extension to determine how to read the file
    std::string::size_type index = filename.find_last_of(".");
    if (index != std::string::npos) {
      bool success = false;
      std::string ext = filename.substr(index);
      if (ext == ".glb") {
        success = tiny.LoadBinaryFromFile(this, &err, filename);
      } else if (ext == ".gltf") {
        success = tiny.LoadASCIIFromFile(this, &err, filename);
      }
      if (!success) {
        throw std::runtime_error(
            "GltfModel:File " + filename + " could not be loaded");
      }
    } else {
      throw std::runtime_error(
          "GltfModel:Filename " + filename + " has no extension");
    }
  }

  /**
   * Return number of meshes, which is actually equal to the number of
   * primitives in the gltf model.
   */
  size_t numPrimitiveMeshes() const;

  /**
   * Add raw data to buffer[0]
   * @param rawData pointer to raw rawData
   * @param numBytes number of bytes to write
   */
  size_t reallyAddToBuffer(const unsigned char* rawData, size_t numBytes);

  /**
   * Adds typed data to buffer[0]
   * @param values a templated vector of values to write
   * @returns {byteOffset, numBytes}
   */
  template <class T>
  std::pair<size_t, size_t> addDataToBuffer(const std::vector<T>& values);

  /**
   * Given data of a particular type, add it to the (assumed unique) buffer,
   * create a buffer view on that data, and add an accessor to access the data.
   * @param values a templated vector of values to write
   * @returns accessor index.
   */
  template <class T>
  int pushData(int target, const std::vector<T>& values);

  /**
   * Add all elements associated with Mesh to the Model.
   * @param mesh a mesh instance
   */
  void addPrimitiveMesh(const Mesh& mesh);

  /**
   * Add all elements associated with TexturedMesh to the Model.
   * @param mesh a textured mesh instance
   */
  void addTexturedPrimitiveMesh(const TexturedMesh& mesh);

  /**
   * Extract mesh with index <meshIndex> and return result in <mesh>.
   * Note the Mesh class has to be allocated first. We do it this way
   * because it plays well with cython.
   * when meshIndex is out of range, an exception is thrown.
   * Note: this a Primitive from gltf, since a Primitive maps to a mesh
   * with a single material, and aligns with our representation of
   * a Mesh and TexturedMesh.
   */
  void extractPrimitiveMesh(const size_t meshIndex, Mesh* mesh) const;

  /**
   * Extract TexturedMesh with index 0, writing properties to output argument.
   * Note that both texcoord and base colors are needed, roughness optional.
   */
  void extractTexturedPrimitiveMesh(const size_t meshIndex, TexturedMesh* mesh)
      const;

  /**
   * Return a list of meshes as a polymorphic list of CMesh* classes or derived.
   * Note, meshes are allocated on the heap and owndership is transferred to
   * caller, who has the responsibility to call delete.
   */
  std::vector<Mesh*> getPolymorphicPrimitiveMeshes() const;

  /**
   * Set URI to sequentially numbered relative image names with given extension.
   * Assumes uris are empty, will throw invalid_argument exception if not.
   */
  void setURIs(const std::string& extension = ".png");

  /**
   * Save all images to given folder, using opencv.
   */
  void saveImages(const std::string& folder) const;

  /**
   * Replace Primitive mesh material
   */
  void updateMaterial(
      size_t meshIndex,
      const Vector3d& color,
      const std::string& uri,
      const std::string& baseDir);

  /**
   * Add a colored material to the model, return the index of the material
   */
  size_t addColoredMaterial(
      const std::string& name,
      const Vector3d& color,
      double metallic = 0.0,
      double roughness = 0.95);

 private:
  // Add an accessor on top of given bufferview
  template <class T>
  int addAccessor(const std::vector<T>& values, int bufferViewIndex);

  // Lowest level access to buffer[0]
  int reallyAddBufferView(
      size_t numBytes,
      int target, // TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER (for indices)|
                  // TINYGLTF_TARGET_ARRAY_BUFFER (for vertex data)
      size_t byteOffset,
      size_t byteStride);

  // Add a bufferView onto buffer[0] memory
  template <class T>
  int addBufferView(size_t numBytes, int target, size_t byteOffset);

  // Add an image and texture based on given a URI, returns texture id.
  // Textures have sequential ids starting at 0.
  int addTexture(const cv::Mat3b& cvImage);

  // Add texture from file to the model
  // uri is file location relative to baseDir. Returns id of the
  // new texture.
  size_t loadTexture(const std::string& uri, const std::string& baseDir);

  // Extract an OpenCV image from texture with given id.
  cv::Mat3b extractTexture(int textureId) const;

  // Get the primitive associated with the given meshIndex
  const tinygltf::Primitive& getPrimitiveMesh(const size_t meshIndex) const;
};

/**
 * Class for getting elements from buffers associated with an object.
 * Accesses scalar values only, and does no bounds checking on access.
 * TODO: fix that, and make working with strides.
 */
template <class T>
class GltfAccessor {
 public:
  /**
   * Create new Accessor
   * @param object a GltfModel instance
   * @param values typed values to calculate min/max values
   * @param bufferViewIndex index to bufferView on which gthis accessor is based
   */
  GltfAccessor(
      const GltfModel& object,
      const std::vector<T>& values,
      int bufferViewIndex);

  /**
   * Create Accessor from existing tinygltf::Accessor
   * @param object a GltfModel instance
   * @param tinyAccessor an existing tinygltf::Accessor instance
   */
  GltfAccessor(const GltfModel& object, const tinygltf::Accessor& tinyAccessor)
      : tinyAccessor_(tinyAccessor) {
    calculatePointer(object);
  }

  // Return number of elements
  size_t size() const {
    return tinyAccessor_.count;
  }

  // Access j^th element from underlying buffer
  T operator[](size_t j) const {
    if (j >= tinyAccessor_.count) {
      throw std::invalid_argument("GltfAccessor[]: invalid element index");
    }
    return pointer_[j];
  }

  // Return all elements in an STL vector
  std::vector<T> asVector() const {
    std::vector<T> result;
    for (size_t i = 0; i < size(); i++) {
      result.push_back((*this)[i]);
    }
    return result;
  }

  // Return underlying tinygltf::Accessor instance
  const tinygltf::Accessor& tinyAccessor() const {
    return tinyAccessor_;
  }

 private:
  // Calculate pointer_, assumes tinyAccessor_ is initialized correctly
  void calculatePointer(const GltfModel& object) {
    if (tinyAccessor_.bufferView >=
        static_cast<int>(object.bufferViews.size())) {
      throw std::invalid_argument("GltfAccessor: invalid bufferView index");
    }
    const tinygltf::BufferView& view =
        object.bufferViews[tinyAccessor_.bufferView];

    if (view.buffer >= static_cast<int>(object.buffers.size())) {
      throw std::invalid_argument("GltfAccessor: invalid buffer index");
    }
    const tinygltf::Buffer& buffer = object.buffers[view.buffer];
    pointer_ = reinterpret_cast<const T*>(
        buffer.data.data() + view.byteOffset + tinyAccessor_.byteOffset);
  }

  tinygltf::Accessor tinyAccessor_; // underlying tinygltf::Accessor
  const T* pointer_; // pointer to associated memory
};

/**
 * Create GltfAccessor mapped onto tinygltf::Accessor with given index <i>
 */
template <class T>
GltfAccessor<T> getAccessor(const GltfModel& object, size_t i) {
  if (i >= object.accessors.size()) {
    throw std::invalid_argument("GltfAccessor: accessor index out of bounds");
  }
  const tinygltf::Accessor& tinyAccessor = object.accessors[i];
  return GltfAccessor<T>(object, tinyAccessor);
}

/**
 * Get type of the component from the accessor.
 */
 int getAccessorType(const GltfModel& object, size_t i);

} // namespace sumo
