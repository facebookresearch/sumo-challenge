/*
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include "Mesh.h"

#define TINYGLTF_NO_STB_IMAGE_WRITE // We are writing our own images
#include "sumo/threedee/tiny_gltf/tiny_gltf.h"

#include <opencv2/core/core.hpp>
#include <iostream>

namespace sumo {

typedef Eigen::Vector2f Vector2f;

/**
 * Textured mesh class which extends Mesh, adding OpenCV textures and texture
 * coordinates. TODO: think about whether we need to support metallic/roughness.
 */
class TexturedMesh : public Mesh {
 public:
  using UVCoord = Vector2f;

 private:
  std::vector<UVCoord> uvCoords_; // texture coordinates, one per vertex
  cv::Mat3b baseColorTexture_, metallicRoughnessTexture_; // OpenCV textures

  friend class GltfModel;

 public:
  /**
   * Construct from indices, vertices, normals, and texture coordinates.
   * In addition, an OpenCV color texture and an optional mettalic roughness
   * texture must/can be provided.
   * Expects:
   *  indices: triplest of indices into vertices.
   *  vertices: vertex positions
   *  normals: normals for those vertices
   *  uvCoords: texture coordinates, one per vertex
   *  baseColorTexture: OpenCV image with texture
   *  metallicRoughnessTexture: *optional* opencv image with metallic/roughness
   */
  TexturedMesh(
      const std::vector<Index>& indices = std::vector<Index>(),
      const std::vector<Vertex>& vertices = std::vector<Vertex>(),
      const std::vector<Normal>& normals = std::vector<Normal>(),
      const std::vector<UVCoord>& uvCoords = std::vector<UVCoord>(),
      const cv::Mat3b& baseColorTexture = cv::Mat3b(),
      const cv::Mat3b& metallicRoughnessTexture = cv::Mat3b());

  // return true if TexturedMesh, needed as cython does not do dynamic casting
  bool isTextured() const override {
    return true;
  }

  // Return reference to uv-coordinates.
  const std::vector<UVCoord>& uvCoords() const {
    return uvCoords_;
  }

  // Return reference to base color texture.
  const cv::Mat3b& baseColorTexture() const {
    return baseColorTexture_;
  }

  // Return reference to metallic roughness texture.
  const cv::Mat3b& metallicRoughnessTexture() const {
    return metallicRoughnessTexture_;
  }

  /**
   * Create a new mesh where a new block of vertices is inserted at the front.
   * Input is <numNewVertices>, how many vertices are inserted, and a list of
   * size n of new indices for the first n old vertices. New vertices are
   * uninitialized unless an old vertex is copied into it. The method does not
   * support renumbering vertices except for a block at the front.
   * NOTE: non-const to avoid problems in cython :-(
   */
  void renumber(size_t numNewVertices, std::vector<Index> vertexRenumbering);

  /**
   * Merge with a second mesh, see base class comments. The base color and
   * metallic_roughness are taken from mesh1. NOTE: non-const to avoid problems
   * in cython :-(
   */
  void merge(const TexturedMesh& mesh2, size_t numCommonVertices = 0);

  /**
  * Replace the geometry of the mesh (vertices, normals, indices, uvCoords) with
  * those of a second mesh.
  */
  void replaceGeometry(const TexturedMesh& mesh2);

  /// Test that the material is dual color/mr texture.
  bool hasDualTextureMaterial() const;
};

} // namespace sumo
