"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


Cython wrapper classes for TinyGLTF class.
"""

cimport numpy as np
import cv2
import numpy as np
import os
import shutil
import tempfile

from sumo.threedee.glb_converter import gltf2glb
from sumo.threedee.gltf_model cimport CGltfModel, GltfModel
from sumo.threedee.box_3d import Box3d
from sumo.threedee.mesh cimport Mesh
from sumo.threedee.textured_mesh cimport TexturedMesh

cdef class TinyGLTF:
  """Internal class used for GltfModel I/O."""

  def __init__(self):
      self.c_ptr = new CTinyGLTF()

  def __dealloc__(self):
      del self.c_ptr

  def load_ascii_from_file(self, unicode path):
      """
      Load instance from gltf file tree.

      Inputs:
      path (string) - Path to input gltf (base directory)

      Return:
      GltfModel instance

      Exceptions
      IOError - if loading fails
      """
      cdef string err
      model = GltfModel()
      cdef CGltfModel* c_model = model.c_ptr
      py_byte_string = path.encode('UTF-8')
      cdef bool ret = self.c_ptr.LoadASCIIFromFile(c_model, &err, py_byte_string)
      if not ret:
          raise IOError("Failed to parse gltf file {}.  Error = {}".format(path, err))
      return model

  def load_binary_from_file(self, unicode path):
      """
      Load GltfModel instance from a glb file.

      Inputs:
      path (string) - Path to input glb file

      Return:
      GltfModel instance

      Exceptions:
      IOError - if loading fails
      """
      cdef string err
      model = GltfModel()
      cdef CGltfModel* c_model = model.c_ptr
      py_byte_string = path.encode('UTF-8')
      cdef bool ret = self.c_ptr.LoadBinaryFromFile(c_model, &err, py_byte_string)
      # set the URI for the images
      c_model.setURIs(b".png")
      if not ret:
          raise IOError("Failed to parse glb file {}.  Error = {}".format(path, err))
      return model

  def write_gltf_scene_to_file(self, GltfModel model, unicode path):
      """Write given GltfModel instance to gltf file at given <path>."""
      cdef CGltfModel* c_model = model.c_ptr
      py_byte_string = path.encode('UTF-8')
      cdef bool ret = self.c_ptr.WriteGltfSceneToFile(c_model, py_byte_string)
      if not ret:
          raise RuntimeError("Failed to write glTF")
