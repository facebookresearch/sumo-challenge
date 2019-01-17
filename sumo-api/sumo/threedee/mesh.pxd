"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


Cython header for interfacing with Mesh C++.
"""

cimport numpy as np
from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.vector cimport vector

from sumo.base.vector cimport CVector3f
from sumo.opencv.wrap cimport Mat3b, Mat1f

cdef extern from "sumo/threedee/Mesh.h" namespace "sumo":
  # Wrap Mesh C++ class
  cdef cppclass CMesh "sumo::Mesh":
    CMesh(const vector[unsigned int]&, const vector[CVector3f]&, const vector[CVector3f]&) except +
    CMesh(const CMesh&) except +
    bool isTextured() const
    size_t numIndices() const
    size_t numVertices() const
    size_t numNormals() const
    const vector[unsigned int]& indices() const
    const vector[CVector3f]& vertices() const
    const vector[CVector3f]& normals() const
    @staticmethod
    vector[CVector3f] CalculateFaceNormals(const vector[unsigned int]&, const vector[CVector3f]&)
    @staticmethod
    vector[CVector3f] EstimateNormals(const vector[unsigned int]&, const vector[CVector3f]&)
    @staticmethod
    CMesh Example(double, bool)
    void cleanupLongEdges(float threshold)
    void cleanupEdgesToOrigin()
    void merge(const CMesh&, size_t)
    void replaceGeometry(const CMesh&)
    bool hasSameMaterial(const CMesh&) const

# Python wrapper class storing a pointer:
cdef class Mesh:
  cdef CMesh* _mesh

# Utility functions. TODO: move to base?
cdef vector[unsigned int] uint32s_of_array(np.ndarray indices)
cdef np.ndarray array_of_uint32s(vector[unsigned int] indices)

cdef Mesh_usurp(CMesh* mesh)
cdef Mesh_copy(CMesh mesh)
