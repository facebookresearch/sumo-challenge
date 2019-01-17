/*
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

// Without this flag, GTSAM Expressions currently fail within fbcode.
// We need EIGEN_MAKE_ALIGNED_OPERATOR_NEW in GTSAM to really fix this
#define EIGEN_DONT_ALIGN_STATICALLY

#include <boost/filesystem.hpp>

#include "Mesh.h"

#include <math.h>

namespace sumo {

using namespace std;

Mesh::Mesh(
    const vector<Index>& indices,
    const vector<Vertex>& vertices,
    const vector<Normal>& normals)
    : indices_(indices), vertices_(vertices), normals_(normals) {
  assert(vertices.size() == normals.size());
}

vector<Mesh::Normal> Mesh::CalculateFaceNormals(
    const vector<Index>& indices,
    const vector<Vertex>& vertices) {
  vector<Normal> faceNormals;
  const size_t numTriangles = indices.size() / 3;
  faceNormals.reserve(numTriangles);
  // Compute face normals using cross product method.
  for (size_t i = 0; i < numTriangles; i++) {
    // get three triangle vertex indices
    const size_t j = i * 3;
    const Index i0 = indices[j], i1 = indices[j + 1], i2 = indices[j + 2];
    Vertex a = vertices[i1] - vertices[i0];
    Vertex b = vertices[i2] - vertices[i0];

    Normal n = a.cross(b).normalized();
    faceNormals.push_back(n);
  }
  return faceNormals;
}

vector<Mesh::Normal> Mesh::EstimateNormals(
    const vector<Index>& indices,
    const vector<Vertex>& vertices) {
  vector<Normal> faceNormals = CalculateFaceNormals(indices, vertices);

  // Initialize normals to zero.
  size_t numVertices = vertices.size();
  vector<Mesh::Normal> normals(numVertices, Normal::Zero());

  // Add face normals to all involved vertices.
  // TODO use iterator
  size_t j = 0;
  for (const auto& n : faceNormals) {
    for (Index i : {indices[j], indices[j + 1], indices[j + 2]}) {
      assert(i < numVertices);
      normals[i] += n;
    }
    j += 3;
  }

  for (auto& n : normals) {
    n.normalize();
  }

  return normals;
}

void Mesh::cleanupLongEdges(float threshold) {
  vector<Index> new_indices;
  new_indices.reserve(numIndices());
  // define distance function
  const auto dist = [](Vertex x, Vertex y) { return (x - y).norm(); };
  // loop over all triangles in this mesh
  for (const auto& triangle : *this) {
    // check all edges
    if (dist(triangle.a, triangle.b) <= threshold &&
        dist(triangle.b, triangle.c) <= threshold &&
        dist(triangle.c, triangle.a) <= threshold) {
      // add triangle since valid
      new_indices.push_back(triangle.i);
      new_indices.push_back(triangle.j);
      new_indices.push_back(triangle.k);
    }
  }
  indices_ = new_indices;
}

void Mesh::cleanupEdgesToOrigin(const double precision) {
  vector<Index> new_indices;
  new_indices.reserve(numIndices());
  // loop over all triangles in this mesh
  for (const auto& triangle : *this) {
    // If any vertex is zero, we bail.
    if (triangle.a.isZero(precision) || triangle.b.isZero(precision) ||
        triangle.c.isZero(precision)) {
      continue;
    }
    // add triangle since valid
    new_indices.push_back(triangle.i);
    new_indices.push_back(triangle.j);
    new_indices.push_back(triangle.k);
  }
  indices_ = new_indices;
}

Mesh Mesh::Example(
    const double length /* = 2.0*/,
    const bool inward /* = true */) {
  vector<Index> indices;
  indices.insert(indices.end(), {1, 0, 2}); // Left face   (x = -1.0f)
  indices.insert(indices.end(), {2, 3, 1}); // Left face   (x = -1.0f)
  indices.insert(indices.end(), {4, 0, 1}); // Bottom face (y = -1.0f)
  indices.insert(indices.end(), {1, 5, 4}); // Bottom face (y = -1.0f)
  indices.insert(indices.end(), {2, 0, 4}); // Back face   (z = -1.0f)
  indices.insert(indices.end(), {4, 6, 2}); // Back face   (z = -1.0f)
  indices.insert(indices.end(), {5, 7, 6}); // Right face  (x = +1.0f)
  indices.insert(indices.end(), {6, 4, 5}); // Right face  (x = +1.0f)
  indices.insert(indices.end(), {6, 7, 3}); // Top face    (y = +1.0f)
  indices.insert(indices.end(), {3, 2, 6}); // Top face    (y = +1.0f)
  indices.insert(indices.end(), {3, 7, 5}); // Front face  (z = +1.0f)
  indices.insert(indices.end(), {5, 1, 3}); // Front face  (z = +1.0f)
  if (inward == false) {
    reverse(indices.begin(), indices.end());
  }

  vector<Vertex> vertices;
  const double radius = length / 2;
  vertices.push_back(Vertex(-1.0f, -1.0f, -1.0f) * radius); // 0th
  vertices.push_back(Vertex(-1.0f, -1.0f, +1.0f) * radius); // 1st
  vertices.push_back(Vertex(-1.0f, +1.0f, -1.0f) * radius); // 2nd
  vertices.push_back(Vertex(-1.0f, +1.0f, +1.0f) * radius); // 3rd
  vertices.push_back(Vertex(+1.0f, -1.0f, -1.0f) * radius); // 4th
  vertices.push_back(Vertex(+1.0f, -1.0f, +1.0f) * radius); // 5th
  vertices.push_back(Vertex(+1.0f, +1.0f, -1.0f) * radius); // 6th
  vertices.push_back(Vertex(+1.0f, +1.0f, +1.0f) * radius); // 7th

  return Mesh(indices, vertices, Mesh::EstimateNormals(indices, vertices));
}

void Mesh::merge(const Mesh& mesh2, size_t numCommonVertices) {
  // Merge the indices.
  const size_t offset = numVertices() - numCommonVertices;
  for (const size_t& index2 : mesh2.indices()) {
    indices_.push_back(index2 < numCommonVertices ? index2 : index2 + offset);
  }

  // Treat the case when mesh2 defines new vertices.
  if (mesh2.numVertices() > numCommonVertices) {
    const auto append = [numCommonVertices](auto& a, const auto& b) {
      a.insert(a.end(), b.begin() + numCommonVertices, b.end());
    };
    append(vertices_, mesh2.vertices());
    append(normals_, mesh2.normals());
  }
}

void Mesh::replaceGeometry(const Mesh& mesh2) {
  //Replace the geometry of the mesh.
  indices_.clear();
  vertices_.clear();
  normals_.clear();

  merge(mesh2, 0);

}

  /// Operator == for Parameter
  // ::: TODO: This function is not correct.  It does not check all aspects
  // of the parameter struct.  Furthermore, some aspects of the Parameter
  // struct are unstable (that is, they are not initialized to known values,
  // so comparing two Parameter objects is not really feasible with the
  // tiny_gltf design.
static bool parametersEqual(
    const tinygltf::Parameter& a,
    const tinygltf::Parameter& b) {
  return
    a.has_number_value == b.has_number_value &&
    a.string_value == b.string_value &&
    a.json_double_value == b.json_double_value;
}

/// Operator == for Parameter
static bool namedParametersEqual(
    const tinygltf::ParameterMap::value_type& a,
    const tinygltf::ParameterMap::value_type& b) {
  return a.first == b.first && parametersEqual(a.second, b.second);
}

/// Operator == for ParameterMap
bool operator==(
    const tinygltf::ParameterMap& a,
    const tinygltf::ParameterMap& b) {
  return a.size() == b.size() &&
      std::equal(a.begin(), a.end(), b.begin(), namedParametersEqual);
}

/// Operator == for Material
// Only comparing ParameterMaps and name.
// TODO: Compare extensions and extras.
bool operator==(const tinygltf::Material& a, const tinygltf::Material& b) {
  return a.additionalValues == b.additionalValues && a.values == b.values &&
      a.name == b.name;
}


bool Mesh::hasSameMaterial(const Mesh& other) const {
  return material_ == other.material_;
}


} // namespace sumo
