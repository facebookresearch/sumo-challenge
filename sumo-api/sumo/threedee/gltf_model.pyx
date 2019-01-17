"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import cv2
import numpy as np
cimport numpy as np
import os
import shutil
import sys
import tempfile

from sumo.base.vector cimport vector3
from sumo.threedee.glb_converter import gltf2glb
from sumo.threedee.gltf cimport TinyGLTF
from sumo.threedee.box_3d import Box3d
from sumo.threedee.mesh cimport Mesh
from sumo.threedee.mesh cimport Mesh, Mesh_usurp
from sumo.threedee.textured_mesh cimport TexturedMesh, TexturedMesh_usurp

cdef class GltfModel:
    """ A wrapper around the C++ GltfModel class, defined in GltfModel.h/cpp.
        Note: also loads and saves all images associated with GLTF object.

        By definition a Gltf model can have multiple Meshes, each with multiple
        primitives. Each primitive is associated with a different material (e.g.
        different texture). Since TexturedMesh class has a single
        material/texture, it maps to a Gltf primitive.
    """

    def __init__(self):
        """Default constructor, just allocates empty C++ struct."""
        self.c_ptr = new CGltfModel()
        self.c_ptr.asset.version = b"2.0"
        self.c_ptr.asset.generator = b"fbsource/fbcode/sumo/threedee/gltf.pyx"

    def __dealloc__(self):
        """Destructor deletes C++ object."""
        del self.c_ptr

    def deepcopy(self):
        """
        Make a copy of self and return it.
        TODO: Figure out how to make this use the deepcopy module interface.  
        
        Return:
        new_model (GltfModel) - copy of self.
        """
        new_model = GltfModel()
        new_model.c_ptr[0] = self.c_ptr[0]
        return new_model
        
    @staticmethod
    def load_from_gltf(unicode path):
        """
        Load instance from gltf file.

        Inputs:
        path (string) - Path to input gltf file

        Return:
        GltfModel instance

        Exceptions
        IOError - if loading fails
        """
        return TinyGLTF().load_ascii_from_file(path)

    @staticmethod
    def load_from_glb(unicode path):
        """
        Load instance from glb file.

        Inputs:
        path (string) - Path to input glb file

        Return:
        GltfModel instance

        Exceptions
        IOError - if loading fails
        """
        return TinyGLTF().load_binary_from_file(path)

    @classmethod
    def from_textured_mesh(cls, TexturedMesh mesh):
        """ Create from single TexturedMesh.
            Keyword arguments:
                mesh (TexturedMesh) - source mesh

        """
        obj = cls()
        obj.add_textured_primitive_mesh(mesh)
        return obj

    @classmethod
    def example(cls):
        """ Example derived from TexturedMesh.example."""
        mesh = TexturedMesh.example()
        return cls.from_textured_mesh(mesh)

    def add_colored_material(self, unicode name, np.ndarray color, double metallic, double roughness):
        """ Add a colored material to the model
        """
        return self.c_ptr.addColoredMaterial(name.encode('UTF-8'), vector3(color), metallic, roughness)


    def update_materials(self, unicode base_dir, list materials):
        """ Update materials in Primitive meshes
            materials is a map describing a the new colored material, or
            None if the Primitive mesh material should not be changed
            NOTE: This mimics the way that suncg handles changing materials
            using the materials defined in the house.json. It is not a
            direct replacement, but and update. If there is a color it will
            update the color, but not change the texture, if there is a texture
            but no color, it will only update the texture.
            Inputs:
              base_dir - uris for textures are defined relative to this
                      base directory
              materials - list of materials. Each material is a dict defining:
                      "color" (Vector3), "uri" (string, empty if this is not
                      a textured material), or None for a material definition
                      that was empty.
        """

        # TODO: Copy and return mutated object copy
        gltf_object = self
        for mesh_index, material in enumerate(materials):
            if mesh_index < gltf_object.num_primitive_meshes():
                if material is not None:
                    color = material["color"]
                    gltf_object.c_ptr.updateMaterial(mesh_index, vector3(color),
                                                     material["uri"].encode('UTF-8'),
                                                     base_dir.encode('UTF-8'))

    def num_buffers(self):
        """Return number of buffers."""
        return self.c_ptr.buffers.size()

    def num_primitive_meshes(self):
        """Return number of primitive meshes."""
        return self.c_ptr.numPrimitiveMeshes()

    def num_nodes(self):
        """Return number of nodes."""
        return self.c_ptr.nodes.size()

    def num_images(self):
        """Return number of images."""
        return self.c_ptr.images.size()

    def image_size(self, size_t i):
        """Return size of image with index <i>."""
        return self.c_ptr.images[i].image.size()

    def image_uri(self, size_t i):
        """Return URI of image with index <i>."""
        bytes = self.c_ptr.images[i].uri
        return bytes.decode('UTF-8')

    def same_size(self, other):
        """Check that both objects have same number of meshes of same size."""
        if other.num_primitive_meshes() != self.num_primitive_meshes():
            return False
        for i in range(self.num_primitive_meshes()):
            my_mesh = self.extract_primitive_mesh(i)
            other_mesh = other.extract_primitive_mesh(i)
            if not my_mesh.same_size(other_mesh):
                return False
        return True

    def add_textured_primitive_mesh(self, TexturedMesh mesh):
        """ Add TexturedMesh to object instance.
            Keyword arguments:
                mesh (TexturedMesh) - source mesh

        """
        self.c_ptr.addTexturedPrimitiveMesh(mesh._textured_mesh[0])

    def add_primitive_mesh(self, Mesh mesh):
        """ Add Mesh to object instance.
            Keyword arguments:
                mesh (Mesh) - source mesh

        """
        self.c_ptr.addPrimitiveMesh(mesh._mesh[0])

    def add_mesh(self, mesh):
        """Polymorphic version of add_primitive_mesh."""
        if isinstance(mesh, TexturedMesh):
            self.add_textured_primitive_mesh(mesh)
        elif isinstance(mesh, Mesh):
            self.add_primitive_mesh(mesh)
        else:
          raise ValueError("GltfModel.add_mesh: expected a Mesh/TexturedMesh")

    def primitive_meshes(self):
        """Return a list of meshes as a polymorphic list."""
        cdef vector[CMesh*] meshes = self.c_ptr.getPolymorphicPrimitiveMeshes()
        return [TexturedMesh_usurp(<CTexturedMesh*>mesh) if mesh.isTextured()
                else Mesh_usurp(mesh)
                for mesh in meshes]

    def extract_primitive_mesh(self, size_t mesh_index=0):
        """Extract mesh with given index, default 0."""
        meshes = self.primitive_meshes()
        return meshes[mesh_index]

    def extract_textured_primitive_mesh(self, size_t mesh_index=0):
        """ Extract textured mesh instance.
            Will fail with Assertionerror if requested is not a TexturedMesh.
        """
        meshes = self.primitive_meshes()
        mesh = meshes[mesh_index]
        assert isinstance(mesh, TexturedMesh)
        return mesh

    def save_as_gltf(self, unicode folder, unicode root):
        """ Save GltfModel instance to GLTF.

            Keyword arguments:
              folder (string) -- folder in which to put the gltf files
              root (string) -- base name for gltf file

            Returns:
              string with full path to gltf file

            Exceptions:
              IOError - if gltf file cannot be written
              RunTimeError - if saveImages fails
        """
        if not os.path.isdir(folder):
            raise ValueError("GltfModel.save_as_gltf: expected a directory")
        gltf_path = os.path.join(folder, root + '.gltf')
        TinyGLTF().write_gltf_scene_to_file(self, gltf_path)
        if not os.path.isfile(gltf_path):
            raise IOError("GLTF conversion failed.")
        self.c_ptr.saveImages(folder.encode('UTF-8'))
        return gltf_path

    def save_as_glb(self, unicode path):
        """ Save GltfModel instance to a GLB file.

            Keyword arguments:
              path (string) -- file name for glb file

            Note, this is a suboptimal hack that saves gltf files in a temporary
            folder and then calls gltf2glb to bundle the files as tinygltf does
            not yet support writing glb files.

            Exceptions:
              IOError - if intermediate gltf file cannot be written
              RunTimeError - if saveImages fails when writing gltf files
        """
        base_name = os.path.basename(path)
        temp = tempfile.mkdtemp()
        temp_folder = temp.decode('UTF-8') if hasattr(temp, 'decode') else temp
        root, extension = os.path.splitext(base_name)
        assert extension == ".glb"
        # Write object to files
        current_dir = os.getcwd()
        os.chdir(temp_folder)
        try:
            gltf_path = self.save_as_gltf(temp_folder, root)
            gltf2glb(gltf_path, path)
        finally:
            shutil.rmtree(temp_folder)
            os.chdir(current_dir)
