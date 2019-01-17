"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import numpy.matlib

from sumo.base.vector cimport vector2fs_of_array, array_of_vector2fs
from sumo.base.vector cimport vector3fs_of_array
from sumo.base.vector import Vector2f, Vector3f
from sumo.opencv.wrap cimport array_from_mat3b
from sumo.opencv.wrap cimport mat3b_from_array
from sumo.threedee.mesh cimport uint32s_of_array

# This is the python wrapper class which dispatches to the C++ class
cdef class TexturedMesh:
    """
      TexturedMesh class: adds base color and metallic roughness textures to a
      regular mesh (in the Mesh class). Adds texture coordinates for mapping.
    """
    def __init__(self, np.ndarray[np.uint32_t, ndim=1] indices = np.ndarray(shape=(0), dtype=np.uint32),
               np.ndarray[np.float32_t, ndim=2] vertices = np.ndarray(shape=(3,0), dtype=np.float32),
               np.ndarray[np.float32_t, ndim=2] normals = np.ndarray(shape=(3,0), dtype=np.float32),
               np.ndarray[np.float32_t, ndim=2] uv_coords = np.ndarray(shape=(2,0), dtype=np.float32),
               np.ndarray[np.uint8_t, ndim=3, mode='c'] base_color_texture = np.zeros([3, 4, 3], dtype=np.uint8),
               np.ndarray[np.uint8_t, ndim=3, mode='c'] metallic_roughness_texture = np.zeros([3, 4, 3], dtype=np.uint8)):

        assert vertices.shape[1] == normals.shape[1]
        # Not all vertices necessarily have uv_coords
        assert vertices.shape[1] >= uv_coords.shape[1]

        cdef vector[unsigned int] cpp_indices = uint32s_of_array(indices)
        cdef vector[CVector3f] cpp_vertices = vector3fs_of_array(vertices)
        cdef vector[CVector3f] cpp_normals = vector3fs_of_array(normals)
        cdef vector[CVector2f] cpp_uv_coords = vector2fs_of_array(uv_coords)
        cdef Mat3b* cv_base_color = mat3b_from_array(base_color_texture)
        cdef Mat3b* cv_metallic_roughness = mat3b_from_array(metallic_roughness_texture)
        self._textured_mesh = new CTexturedMesh(cpp_indices, cpp_vertices, cpp_normals,
                                                cpp_uv_coords,
                                                cv_base_color[0],
                                                cv_metallic_roughness[0])
        self._mesh = self._textured_mesh
        del cv_base_color
        del cv_metallic_roughness

    def __dealloc__(self):
        # del self._textured_mesh
        pass

    @classmethod
    def from_mesh(cls, mesh, color=np.full((3, 1), 255, dtype=np.uint8)):
        """
        Create a TexturedMesh from a Mesh.  Generates a small, uniform texture for
        the base color and no texture for the metallic roughness.  Texture coordinates
        are all set to 0.

        Inputs:
        mesh (Mesh)
        color (3x1 np.array of uint8) - color of texture (default is white)

        Return:
        TexturedMesh
        """
        uv_coords = np.zeros((2, mesh.num_vertices()), dtype=np.float32)
        base_color = np.reshape(np.resize(color, (4, 3)), (2, 2, 3))
        return TexturedMesh(mesh.indices(), mesh.vertices(), mesh.normals(),
                            uv_coords, base_color)

    @classmethod
    def cube(cls,
             np.ndarray[np.uint8_t, ndim=3, mode='c'] base_color_texture,
             np.ndarray[np.uint8_t, ndim=3, mode='c'] metallic_roughness_texture):
        """ Create a 3D mesh of 12 triangles.
            The textures is assumed to be laid out in cube-map format, i.e.,
            a 1:6 aspect ratio with BACK, LEFT, FRONT, RIGHT, UP, DOWN order.
            # TODO: make general to have m triangle strips of 2*n triangles
        """
        # Create 2D mesh
        cdef size_t j = 0
        cdef int size = base_color_texture.shape[0]
        assert base_color_texture.shape[1] == 6 * size
        indices = np.empty((36,), dtype=np.uint32)
        pixels = np.empty((2,24),dtype=np.float32)
        for i in range(6): # BACK, LEFT, FRONT, RIGHT, UP, DOWN
            offset = Vector2f(i*size,0)
            pixels[:,j+0] = offset + Vector2f(   0, size) # bottom-left
            pixels[:,j+1] = offset + Vector2f(size, size) # bottom-right
            pixels[:,j+2] = offset + Vector2f(   0,    0) # top-left
            pixels[:,j+3] = offset + Vector2f(size,    0) # top-right
            indices[i*6:i*6+6] = [j+0, j+1, j+2,  j+1, j+3, j+2]
            j+=4

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

        # Convert pixels to uv_coords in RGB image
        uv_coords = pixels/size
        uv_coords[0,:] = uv_coords[0,:]/6.0
        uv_coords[1,:] = uv_coords[1,:]

        return cls(indices, vertices, normals, uv_coords,
                   base_color_texture, metallic_roughness_texture)

    @classmethod
    def example(cls):
       """Create a textured mesh to test with."""
       base_color = np.empty((4, 24, 3), dtype=np.uint8)
       metallic_roughness = np.empty((2, 12, 3), dtype=np.uint8)
       for c in range(3):
           base_color[:, :, c] = c
           metallic_roughness[:, :, c] = c
       return cls.cube(base_color, metallic_roughness)

    def uv_coords(self):
        return array_of_vector2fs(self._textured_mesh.uvCoords())

    def base_color(self):
        cdef const Mat3b* mat = &self._textured_mesh.baseColorTexture()
        return array_from_mat3b(mat[0])

    def metallic_roughness(self):
        cdef const Mat3b* mat = &self._textured_mesh.metallicRoughnessTexture()
        return array_from_mat3b(mat[0])

    def rotate(self, R):
            """Return rotated mesh, given Rot3 instance <R>."""
            return TexturedMesh(self.indices(),
                                (R * self.vertices()).astype(np.float32),
                                (R * self.normals()).astype(np.float32),
                                self.uv_coords(),
                                self.base_color(),
                                self.metallic_roughness())

    def transform(self, T):
            """Return transformed textured mesh, given Pose3 instance <T>."""
            return TexturedMesh(self.indices(),
                                (T * self.vertices()).astype(np.float32),
			        (T.R * self.normals()).astype(np.float32),
		                self.uv_coords(),	
                                self.base_color(),
                                self.metallic_roughness())

    def renumber(self, size_t num_new_vertices, list vertex_renumbering):
        """Wrapper around CTexturedMesh::renumber."""
        # vertex_renumbering is automatically converted to vector[unsigned int]
        self._textured_mesh.renumber(num_new_vertices,vertex_renumbering)

    def merge(self, TexturedMesh mesh2, size_t num_common_vertices=0):
        """Wrapper around CTexturedMesh::merge."""
        self._textured_mesh.merge(mesh2._textured_mesh[0],num_common_vertices)

    def replace_geometry(self, TexturedMesh mesh2):
      """Wrapper around CTexturedMesh::replaceGeometry."""
      self._textured_mesh.replaceGeometry(mesh2._textured_mesh[0])

    def has_dual_texture_material(self):
        """Test that the material is dual color/mr texture."""
        return self._textured_mesh.hasDualTextureMaterial()

    @staticmethod
    def merge_quadrant_meshes(mesh0, mesh1, mesh2, mesh3):
        """ Merge 4 quads that are result of subdivision with this numbering scheme:
              2 7 3
              4 6 8
              0 5 1
            The 4 input meshes are assumed to be the quadrants that cointain
            vertices 0, 1, 2, and 3, respectively, in that order.
            NOTE: mutates arguments, takes name from mesh0
        """
        # TODO: is this too complicated? Maybe a better merge obviates renumbering.
        # merge bottom quads 0,5,4,6 and 5,1,6,8
        mesh0.renumber(4, [2,0,3,1]) # 0,5,4,6 -> 5,6,0,4
        mesh1.renumber(4, [0,2,1,3]) # 5,1,6,8 -> 5,6,1,8
        mesh0.merge(mesh1, 2) # 5,6,0,4,1,8
        # merge top quads 4,6,2,7 and  6,7,2,4
        mesh2.renumber(4, [3,0,2,1]) # 4,6,2,7 -> 6,7,2,4
        mesh3.renumber(4, [0,3,1,2]) # 6,8,7,3 -> 6,7,3,8
        mesh2.merge(mesh3, 2) # 6,7,2,4,3,8
        # merge top and bottom
        mesh0.renumber(6, [4,1,3,0,5,2]) # 4,6,8,0,5,1
        mesh2.renumber(6, [1,4,3,0,5,2]) # 4,6,8,2,7,3
        mesh0.merge(mesh2, 3) # 4,6,8,0,5,1,2,7,3
        # renumber
        mesh0.renumber(9, [4,6,8,0,5,1,2,7,3])
        return mesh0

cdef TexturedMesh_usurp(CTexturedMesh* mesh):
    """Create a TexturedMesh class pointing to C++ class on heap with given pointer."""
    wrapper = <TexturedMesh>TexturedMesh.__new__(TexturedMesh)
    del wrapper._textured_mesh
    wrapper._textured_mesh = mesh
    wrapper._mesh = mesh
    return wrapper

cdef TexturedMesh_copy(CTexturedMesh mesh):
    """Create a TexturedMesh class pointing to C++ class copied from C++ instance."""
    # TODO: do something better than copying the mesh here. Same as
    # colored_mesh.pyx
    return TexturedMesh_usurp(new CTexturedMesh(mesh))
