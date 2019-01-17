"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


Cython header for interfacing with TinyGltf C++ library.
"""

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

from sumo.threedee.mesh cimport CMesh
from sumo.threedee.textured_mesh cimport CTexturedMesh
from sumo.threedee.gltf_model cimport CGltfModel

# Import C++ into cython:

cdef extern from "sumo/threedee/tiny_gltf/tiny_gltf.h" namespace "tinygltf":
  cdef cppclass CAsset "tinygltf::Asset":
    string version
    string generator
    CAsset() except +

  cdef cppclass CBuffer "tinygltf::Buffer":
    CBuffer() except +

  cdef cppclass CGltfMesh "tinygltf::Mesh":
    CGltfMesh() except +

  cdef cppclass CImage "tinygltf::Image":
    CImage() except +
    string uri
    int width
    int height
    vector[unsigned char] image

  cdef cppclass CNode "tinygltf::Node":
    CNode() except +

  cdef cppclass CTinyGLTF "tinygltf::TinyGLTF":
    CTinyGLTF() except +
    bool LoadASCIIFromFile(CGltfModel* model, string* err,
                           const string& filename)
    bool WriteGltfSceneToFile(CGltfModel* model, const string& filename)

    bool LoadBinaryFromFile(
      CGltfModel* model,
      string* err,
      const string& filename)


# Python wrapper classes:

cdef class TinyGLTF:
  cdef CTinyGLTF* c_ptr
