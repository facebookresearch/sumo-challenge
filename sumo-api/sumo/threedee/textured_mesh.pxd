"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from sumo.base.vector cimport CVector2f, CVector3f
from sumo.opencv.wrap cimport Mat3b
from sumo.threedee.mesh cimport CMesh, Mesh

cdef extern from "sumo/threedee/TexturedMesh.h" namespace "sumo":
  # Wrap TexturedMesh C++ class
  cdef cppclass CTexturedMesh "sumo::TexturedMesh" (CMesh):
    CTexturedMesh()
    CTexturedMesh(const vector[unsigned int]&, const vector[CVector3f]&,
                  const vector[CVector3f]&, const vector[CVector2f]&,
                  const Mat3b&, const Mat3b&) except +
    CTexturedMesh(const CTexturedMesh&)
    const vector[CVector2f]& uvCoords() const
    const Mat3b& baseColorTexture() const
    const Mat3b& metallicRoughnessTexture() const
    void renumber(size_t, vector[unsigned int])
    void merge(const CTexturedMesh&, size_t)
    void replaceGeometry(const CTexturedMesh&)
    bool hasDualTextureMaterial() const

# Python wrapper class storing a pointer:
cdef class TexturedMesh(Mesh):
  cdef CTexturedMesh* _textured_mesh

cdef TexturedMesh_usurp(CTexturedMesh* mesh)
cdef TexturedMesh_copy(CTexturedMesh)
