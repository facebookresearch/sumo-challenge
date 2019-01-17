/*
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <opencv2/core/core.hpp>

#define TINYGLTF_NO_STB_IMAGE_WRITE // We are writing our own images
#include "sumo/threedee/tiny_gltf/tiny_gltf.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

namespace sumo {

typedef Eigen::Vector3f Vector3f;

/**
 * Mesh class, no texture and color.
 */
class Mesh {
 public:
  using Index = unsigned int;
  using Vertex = Vector3f;
  using Normal = Vector3f;
  using Color = Vector3f;

  struct triangle {
    triangle(const Index i, const Index j, const Index k, const Mesh& mesh)
        : i(i),
          j(j),
          k(k),
          a(mesh.vertices()[i]),
          b(mesh.vertices()[j]),
          c(mesh.vertices()[k]) {}
    const Index i, j, k;
    const Vertex &a, &b, &c;
  };

  /**
   * iterator class nested in Mesh class to iterate all triangles in it.
   */
  struct triangle_iterator {
    explicit triangle_iterator(const Mesh& parent) : index_(0), mesh_(parent) {}
    triangle_iterator(const Mesh& parent, size_t v)
        : index_(v), mesh_(parent) {}
    ~triangle_iterator() {}

    triangle operator*() const {
      const auto& indices = mesh_.indices();
      Index i = indices[index_];
      Index j = indices[index_ + 1];
      Index k = indices[index_ + 2];
      return triangle(i, j, k, mesh_);
    }

    triangle_iterator operator++(int) /* postfix */ {
      return triangle_iterator(mesh_, index_ + 3);
    }
    triangle_iterator& operator++() /* prefix */ {
      index_ += 3;
      return *this;
    }
    bool operator==(const triangle_iterator& rhs) const {
      return index_ == rhs.index_;
    }
    bool operator!=(const triangle_iterator& rhs) const {
      return index_ != rhs.index_;
    }


   protected:
    size_t index_;
    const Mesh& mesh_;
  };

  triangle_iterator begin() const {
    return triangle_iterator(*this);
  }

  triangle_iterator end() const {
    return triangle_iterator(*this, 3 * numTriangles());
  }

  /**
   * Construct from indices, vertices, and normals.
   * Inputs:
   *  indices: triplets of indices into vertices.
   *  vertices: vertex positions
   *  normals: normals for those vertices
   */
  explicit Mesh(
      const std::vector<Index>& indices = std::vector<Index>(),
      const std::vector<Vertex>& vertices = std::vector<Vertex>(),
      const std::vector<Normal>& normals = std::vector<Normal>());

  Mesh(const Mesh& other)
      : indices_(other.indices()),
        vertices_(other.vertices()),
        normals_(other.normals()) {}

  virtual ~Mesh() {}

  // return true if TexturedMesh, needed as cython does not do dynamic casting
  virtual bool isTextured() const {
    return false;
  }

  // number of indices
  size_t numIndices() const {
    return indices_.size();
  }

  // number of triangles
  size_t numTriangles() const {
    return indices_.size() / 3;
  }

  // number of vertices
  size_t numVertices() const {
    return vertices_.size();
  }

  // Return reference to indices.
  const std::vector<Index>& indices() const {
    return indices_;
  }

  // Return reference to vertices.
  const std::vector<Vertex>& vertices() const {
    return vertices_;
  }

  // Return reference to normals.
  const std::vector<Normal>& normals() const {
    return normals_;
  }

  /**
   * Construct an axis-aligned uncolored cube with the given side length
   * centered at the origin of the mesh coordinates
   * Inputs:
   *    length: side length of the cube
   *    inward: true if the cube has inward surface normal vectors
   * Output:
   *    Mesh instance of the cube
   */
  static Mesh Example(const double length = 2.0, const bool inward = false);

  /**
   * Calculate face normals.
   * Inputs:
   *  indices: triplets of indices into vertices.
   *  vertices: vertex positions
   */
  static std::vector<Normal> CalculateFaceNormals(
      const std::vector<Index>& indices,
      const std::vector<Vertex>& vertices);

  /**
   * Estimate vertex normals.
   * Inputs:
   *  indices: triplets of indices into vertices.
   *  vertices: vertex positions
   */
  static std::vector<Normal> EstimateNormals(
      const std::vector<Index>& indices,
      const std::vector<Vertex>& vertices);

  /**
   * Remove triangles with long edges.
   * Keyword arguments:
   *     threshold -- removes triangles with any edge longer than this.
   * NOTE: non-const to avoid problems in cython :-(
   */
  void cleanupLongEdges(float threshold);

  /**
   * Remove all triangles connecting to (0,0,0). Does not remove the vertices.
   * Keyword arguments:
   *     precision -- precision argument to Eigen isZero()
   NOTE: non-const to avoid problems in cython :-(
   */
  void cleanupEdgesToOrigin(const double precision = 1e-10);

  /**
   * Merge with a second mesh, assuming <numCommonVertices> vertices in common,
   * that are assumed to be numbered 0...numCommonVertices-1. The first mesh can
   * define more vertices, and indices that refer to them will remain unchanged.
   * If the second mesh defines more vertices, they will be re-numbered to come
   * after the first mesh vertices. NOTE: non-const to avoid problems in cython
   */
  void merge(const Mesh& mesh2, size_t numCommonVertices = 0);

  /**
  * Replace the geometry of the mesh (vertices, normals, indices) with those of
  * a second mesh.
  */
  void replaceGeometry(const Mesh& mesh2);

  /// Check whether material properties are equal.
  bool hasSameMaterial(const Mesh& other) const;

 protected:
  std::vector<Index> indices_; // triplets of vertex indices
  std::vector<Vertex> vertices_; // 3D vertex positions
  std::vector<Normal> normals_; // normals for those vertices
  tinygltf::Material material_; // Gltf compatible material properties
  friend class GltfModel;
};

} // namespace sumo
