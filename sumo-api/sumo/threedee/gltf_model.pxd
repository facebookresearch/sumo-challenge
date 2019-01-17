"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


Cython header for interfacing with TinyGltf C++ library.
"""

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

from sumo.base.vector cimport CVector3
from sumo.threedee.gltf cimport CAsset, CBuffer, CGltfMesh, CNode, CImage
from sumo.threedee.mesh cimport CMesh
from sumo.threedee.textured_mesh cimport CTexturedMesh

# Import C++ into cython:

cdef extern from "sumo/threedee/GltfModel.h" namespace "sumo":
  cdef cppclass CGltfModel "sumo::GltfModel":
    CAsset asset
    size_t numPrimitiveMeshes()
    vector[CBuffer] buffers
    vector[CMesh] meshes
    vector[CNode] nodes
    vector[CImage] images
    vector[CMesh*] getPolymorphicPrimitiveMeshes() except +
    CGltfModel() except +
    void addTexturedPrimitiveMesh(const CTexturedMesh& mesh)
    void addPrimitiveMesh(const CMesh& mesh)
    void updateMaterial(size_t meshIndex, CVector3& color, string& uri,
                        string& baseDir)
    size_t addColoredMaterial(string& name, CVector3& color, double metallic, double roughness)
    void setURIs(const string& extension) except +
    void saveImages(const string& folder) except +

# Python wrapper classes:

cdef class GltfModel:
  cdef CGltfModel* c_ptr
