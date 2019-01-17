"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import numpy.matlib

from sumo.base.vector cimport vector3fs_of_array, array_of_vector3fs
from sumo.base.vector import Vector2f, Vector3f
from sumo.opencv.wrap cimport (
    Mat3b, Mat1f, array_from_mat3b, mat3b_from_array, array_from_mat1f
)

cdef vector[unsigned int] uint32s_of_array(np.ndarray indices):
  cdef vector[unsigned int] cpp_indices
  for i in range(indices.shape[0]):
    cpp_indices.push_back(<unsigned int>indices[i])
  return cpp_indices

cdef np.ndarray array_of_uint32s(vector[unsigned int] indices):
  cdef size_t n = indices.size()
  cdef np.ndarray result = np.empty((n, ), dtype=np.uint32)
  for i in range(n):
    result[i] = indices[i]
  return result

# This is the python wrapper class which dispatches to the C++ class
cdef class Mesh:
  """A simple mesh class."""
  def __init__(self, np.ndarray[np.uint32_t, ndim=1] indices = np.ndarray(shape=(0), dtype=np.uint32),
               np.ndarray[np.float32_t, ndim=2] vertices = np.ndarray(shape=(3,0), dtype=np.float32),
               np.ndarray[np.float32_t, ndim=2] normals = np.ndarray(shape=(3,0), dtype=np.float32)):
    assert vertices.shape[1] == normals.shape[1]
    cdef vector[unsigned int] cpp_indices = uint32s_of_array(indices)
    cdef vector[CVector3f] cpp_vertices = vector3fs_of_array(vertices)
    cdef vector[CVector3f] cpp_normals = vector3fs_of_array(normals)
    self._mesh = new CMesh(cpp_indices, cpp_vertices, cpp_normals)

  def __dealloc__(self):
    del self._mesh
    pass

  def is_textured(self):
    return self._mesh.isTextured()

  def indices(self):
    return array_of_uint32s(self._mesh.indices())

  def vertices(self):
    return array_of_vector3fs(self._mesh.vertices())

  def normals(self):
    return array_of_vector3fs(self._mesh.normals())

  # TODO: make it a property
  def num_indices(self):
    return self._mesh.numIndices()

  # TODO: make it a property
  def num_vertices(self):
    return self._mesh.numVertices()

  def same_size(self, other):
      """Check that two meshes have same size."""
      return self.num_indices() == other.num_indices() and self.num_vertices() == other.num_vertices()

  @classmethod
  def example(cls, const double length = 2, bool inward = False):
      """ Create a Mesh example of a axis-aligned cube.
      Inputs:
          length -- side length of the cube
          inward -- true if the cube has inward surface normal vectors
      Return:
          Mesh instance
      """
      return Mesh_copy(CMesh.Example(length, inward))

  def rotate(self, R):
      """Return rotated mesh, given Rot3 instance <R>."""
      return Mesh(self.indices(),
                  (R * self.vertices()).astype(np.float32),
                  (R * self.normals()).astype(np.float32))

  def transform(self, T):
      """Return transformed mesh, given Pose3 instance <T>."""
      return Mesh(self.indices(),
                  (T * self.vertices()).astype(np.float32),
                  (T.R * self.normals()).astype(np.float32))

  @staticmethod
  def calculate_face_normals(np.ndarray[np.uint32_t, ndim=1] indices, np.ndarray[np.float32_t, ndim=2] vertices):
    cdef vector[unsigned int] cpp_indices = uint32s_of_array(indices)
    cdef vector[CVector3f] cpp_vertices = vector3fs_of_array(vertices)
    return array_of_vector3fs(CMesh.CalculateFaceNormals(cpp_indices, cpp_vertices))

  @classmethod
  def from_open3d_mesh(cls, open3d_mesh):
      """ Create a Mesh from a Open3D Mesh having the OpenGL coordinate system
      Inputs:
          mesh (Open3D Mesh)
      Return:
          Mesh
      """
      # Extract indices, vertices, normal
      indices = np.asarray(open3d_mesh.triangles).astype(np.uint32).flatten()
      vertices = np.asarray(open3d_mesh.vertices).astype(np.float32).T
      if len(open3d_mesh.vertex_normals) != len(open3d_mesh.vertices):
          open3d_mesh.compute_vertex_normals()
      normals = np.asarray(open3d_mesh.vertex_normals).astype(np.float32).T
      return cls(indices, vertices, normals)

  @classmethod
  def cube(cls):
      """ Create a 3D mesh of 12 triangles.
          The textures is assumed to be laid out in cube-map format, i.e.,
          a 1:6 aspect ratio with BACK, LEFT, FRONT, RIGHT, UP, DOWN order.
          # TODO: make general to have m triangle strips of 2*n triangles
      """
      # Create 2D mesh
      cdef size_t j = 0
      indices = np.empty((36,), dtype=np.uint32)

      # Create 3D mesh in OpenGL frame, Y is up, positive Z is back
      # Face order is BACK, LEFT, FRONT, RIGHT, UP, DOWN
      vertices = np.array(
        [[ 1, -1,  1, -1,  -1, -1, -1, -1,  -1,  1, -1,  1,   1,  1,  1,  1,  -1,  1, -1,  1,  -1,  1, -1,  1],
         [-1, -1,  1,  1,  -1, -1,  1,  1,  -1, -1,  1,  1,  -1, -1,  1,  1,   1,  1,  1,  1,  -1, -1, -1, -1],
         [ 1,  1,  1,  1,   1, -1,  1, -1,  -1, -1, -1, -1,  -1,  1, -1,  1,  -1, -1,  1,  1,   1,  1, -1, -1]],
        dtype=np.float32)

      # Create inward normals, in OpenGL frame, Y is up, positive Z is back
      # normals for BACK, LEFT, FRONT, RIGHT, UP, DOWN
      dir = [(0,0,-1),(1,0,0),(0,0,1),(-1,0,0),(0,-1,0),(0,1,0)]
      normals = np.empty((3,24),dtype=np.float32)
      for j in range(6):
        normals[:,j*4:j*4+4] = np.column_stack([Vector3f(*dir[j])] * 4)

      return cls(indices, vertices, normals)

  @staticmethod
  def estimate_normals(np.ndarray[np.uint32_t, ndim=1] indices, np.ndarray[np.float32_t, ndim=2] vertices):
    cdef vector[unsigned int] cpp_indices = uint32s_of_array(indices)
    cdef vector[CVector3f] cpp_vertices = vector3fs_of_array(vertices)
    return array_of_vector3fs(CMesh.EstimateNormals(cpp_indices, cpp_vertices))

  def cleanup_long_edges(self, np.float32_t threshold = 5):
    self._mesh.cleanupLongEdges(threshold)

  def cleanup_edges_to_origin(self):
    self._mesh.cleanupEdgesToOrigin()

  def merge(self, Mesh mesh2, size_t num_common_vertices=0):
      """Wrapper around CMesh::merge."""
      self._mesh.merge(mesh2._mesh[0],num_common_vertices)

  def replace_geometry(self, Mesh mesh2):
      """Wrapper around CMesh::replaceGeometry."""
      self._mesh.replaceGeometry(mesh2._mesh[0])

  def has_same_material(self, Mesh other):
      """Check whether material properties are equal."""
      return self._mesh.hasSameMaterial(other._mesh[0])

cdef Mesh_usurp(CMesh* mesh):
    """Create a Mesh class pointing to C++ class on heap with given pointer."""
    wrapper = <Mesh>Mesh.__new__(Mesh)
    del wrapper._mesh
    wrapper._mesh = mesh
    return wrapper

cdef Mesh_copy(CMesh mesh):
    """Create a Mesh class pointing to C++ class copied from C++ instance."""
    # TODO: not loving that we copy below, but don't know how else to do it.
    return Mesh_usurp(new CMesh(mesh))
