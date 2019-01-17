#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


TexturedMesh class unit tests.
"""


import shutil
import tempfile
import unittest

import numpy as np
from sumo.base.vector import Vector2f, Vector3f
from sumo.threedee.box_3d import Box3d
from sumo.threedee.textured_mesh import TexturedMesh


class TestTexturedMesh(unittest.TestCase):
    def setUp(self):
        rows, cols = 3, 4
        self.base_color = np.empty((rows, cols, 3), dtype=np.uint8)
        self.metallic_roughness = np.empty((rows, cols, 3), dtype=np.uint8)
        self.vertices = np.column_stack(
            [Vector3f(0, 0, 0), Vector3f(1, 0, 0), Vector3f(0, 1, 0), Vector3f(1, 1, 0)]
        )
        self.normals = np.column_stack([Vector3f(0, 0, 1)] * 4)
        self.uv_coords = np.column_stack(
            [Vector2f(0, 0), Vector2f(1, 0), Vector2f(0, 1), Vector2f(1, 1)]
        )
        self.temp_directory = tempfile.mkdtemp()
        self.indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)
        self.mesh = TexturedMesh(
            self.indices,
            self.vertices,
            self.normals,
            self.uv_coords,
            self.base_color,
            self.metallic_roughness,
        )

    def tearDown(self):
        """Clean up: remove temporary outout directory."""
        shutil.rmtree(self.temp_directory)

    def test_constructor(self):
        """Create a minimal example and test constructor with it."""
        self.assertEqual(self.mesh.num_indices(), 6)
        self.assertEqual(self.mesh.num_vertices(), 4)
        np.testing.assert_array_equal(self.mesh.indices(), self.indices)
        np.testing.assert_array_equal(self.mesh.vertices(), self.vertices)
        np.testing.assert_array_equal(self.mesh.normals(), self.normals)
        np.testing.assert_array_equal(self.mesh.uv_coords(), self.uv_coords)
        self.assertTrue(self.mesh.has_dual_texture_material())

    def test_merge(self):
        """Test merging two meshes with common vertices."""
        vertices = np.column_stack([self.vertices, self.vertices])
        self.assertEqual(vertices.shape, (3, 8))
        normals = np.column_stack([self.normals, self.normals])
        uv_coords = np.column_stack([self.uv_coords, self.uv_coords])

        indices1 = np.array([0, 1, 2, 4, 5, 7], dtype=np.uint32)
        mesh = TexturedMesh(
            indices1,
            vertices,
            normals,
            uv_coords,
            self.base_color,
            self.metallic_roughness,
        )

        indices2 = np.array([1, 3, 2, 4, 5, 7], dtype=np.uint32)
        mesh2 = TexturedMesh(
            indices2,
            vertices,
            normals,
            uv_coords,
            self.base_color,
            self.metallic_roughness,
        )

        mesh.merge(mesh2, 4)
        self.assertEqual(mesh.vertices().shape, (3, 12))
        expected = np.column_stack([self.vertices, self.vertices, self.vertices])
        np.testing.assert_array_equal(mesh.vertices(), expected)
        expected_indices = np.array(
            [0, 1, 2, 4, 5, 7, 1, 3, 2, 8, 9, 11], dtype=np.uint32
        )
        np.testing.assert_array_equal(mesh.indices(), expected_indices)

    def test_renumber(self):
        """Test renumbering some of the vertices."""
        mesh = self.mesh
        mesh.renumber(6, [5, 0])
        # 6 to make room for new indices, 4 for old indices, - 2 renumbered
        # index 1 is renumbered to 0
        # index 0 is renumbered to 5
        self.assertEqual(mesh.num_indices(), 6)
        expected_indices = np.array([5, 0, 2 + 6, 0, 3 + 6, 2 + 6], dtype=np.uint32)
        np.testing.assert_array_equal(mesh.indices(), expected_indices)
        self.assertEqual(mesh.num_vertices(), 6 + 4 - 2)
        np.testing.assert_array_equal(mesh.vertices()[:, 0], self.vertices[:, 1])
        np.testing.assert_array_equal(mesh.normals()[:, 5], self.normals[:, 0])
        np.testing.assert_array_equal(mesh.uv_coords()[:, 5], self.uv_coords[:, 0])

    def test_merge_quadrant_meshes(self):
        """Test merging 4 quads that (could be) result of subdivision."""
        indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)
        # 2 7 3
        # 4 6 8
        # 0 5 1
        vertices = np.column_stack(
            [
                Vector3f(0, 0, 0),
                Vector3f(1, 1, 1),
                Vector3f(2, 2, 2),
                Vector3f(3, 3, 3),
                Vector3f(4, 4, 4),
                Vector3f(5, 5, 5),
                Vector3f(6, 6, 6),
                Vector3f(7, 7, 7),
                Vector3f(8, 8, 8),
            ]
        )
        args = self.normals, self.uv_coords, self.base_color, self.metallic_roughness
        mesh0 = TexturedMesh(indices, vertices[:, [0, 5, 4, 6]], *args)
        mesh1 = TexturedMesh(indices, vertices[:, [5, 1, 6, 8]], *args)
        mesh2 = TexturedMesh(indices, vertices[:, [4, 6, 2, 7]], *args)
        mesh3 = TexturedMesh(indices, vertices[:, [6, 8, 7, 3]], *args)
        mesh = TexturedMesh.merge_quadrant_meshes(mesh0, mesh1, mesh2, mesh3)
        self.assertEqual(mesh.num_indices(), 4 * 3 * 2)
        self.assertEqual(mesh.num_vertices(), 9)
        expected_indices = (
            [0, 5, 4, 5, 6, 4]
            + [5, 1, 6, 1, 8, 6]
            + [4, 6, 2, 6, 7, 2]
            + [6, 8, 7, 8, 3, 7]
        )
        np.testing.assert_array_equal(mesh.indices(), expected_indices)
        np.testing.assert_array_equal(mesh.vertices(), vertices)

    def test_cube(self):
        """Check generation of a cube mesh."""
        rows, cols = 4, 24
        base_color = np.empty((rows, cols, 3), dtype=np.uint8)
        metallic_roughness = np.empty((rows, cols, 3), dtype=np.uint8)
        mesh = TexturedMesh.cube(base_color, metallic_roughness)

        # Check indices
        indices = mesh.indices()
        self.assertEqual(indices.shape, (6 * 6,))
        np.testing.assert_array_equal(indices[:6], [0, 1, 2, 1, 3, 2])
        np.testing.assert_array_equal(indices[-6:], [20, 21, 22, 21, 23, 22])

        # Check 3D mesh
        vertices = mesh.vertices()
        self.assertEqual(vertices.shape, (3, 4 * 6))
        # Check BACK, where y = -1
        np.testing.assert_array_equal(
            vertices[:, :4],
            np.column_stack(
                [
                    Vector3f(+1, -1, +1),
                    Vector3f(-1, -1, +1),
                    Vector3f(+1, +1, +1),
                    Vector3f(-1, +1, +1),
                ]
            ),
        )
        # Check LEFT, FRONT, RIGHT, UP
        np.testing.assert_array_equal(vertices[0, 4:8], (-1, -1, -1, -1))
        np.testing.assert_array_equal(vertices[2, 8:12], (-1, -1, -1, -1))
        np.testing.assert_array_equal(vertices[0, 12:16], (1, 1, 1, 1))
        np.testing.assert_array_equal(vertices[1, 16:20], (1, 1, 1, 1))
        # Check DOWN, where Z = -1
        np.testing.assert_array_equal(
            vertices[:, -4:],
            np.column_stack(
                [
                    Vector3f(-1, -1, +1),
                    Vector3f(+1, -1, +1),
                    Vector3f(-1, -1, -1),
                    Vector3f(+1, -1, -1),
                ]
            ),
        )

        # check cpp_normals
        normals = mesh.normals()
        # normals for BACK, LEFT, FRONT, RIGHT, UP, DOWN
        dir = [(0, 0, -1), (1, 0, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 1, 0)]
        for j in range(6):
            np.testing.assert_array_equal(normals[:, j * 4], dir[j])

        # Check texture coordinates
        uv_coords = mesh.uv_coords()
        np.testing.assert_array_almost_equal(
            uv_coords[:, :4], [[0, 1 / 6.0, 0, 1 / 6.0], [1, 1, 0, 0]]
        )
        np.testing.assert_array_equal(np.min(uv_coords, axis=1), [0, 0])
        np.testing.assert_array_equal(np.max(uv_coords, axis=1), [1, 1])

    def test_merge6(self):
        """Test merging 6 meshes."""
        # Create 6 separate meshes
        # loop over all cube faces
        faces = []
        for _ in range(6):
            # TODO: faces.append(self.mesh) bombs, probably memory alloc issues
            faces.append(
                TexturedMesh(
                    self.indices,
                    self.vertices,
                    self.normals,
                    self.uv_coords,
                    self.base_color,
                    self.metallic_roughness,
                )
            )
        # merge faces:
        mesh = faces[5]
        for i in range(5):
            mesh.merge(faces[i], 0)
        self.assertEqual(mesh.num_indices(), 36)
        self.assertEqual(mesh.num_vertices(), 24)

    def test_from_mesh(self):
        """Test converting mesh to textured mesh."""
        mesh = Box3d([0, 0, 0], [1, 1, 1]).to_mesh()
        color = np.array([255, 0, 0], dtype=np.uint8)  # red
        textured_mesh = TexturedMesh.from_mesh(mesh, color)

        self.assertIsInstance(textured_mesh, TexturedMesh)
        self.assertEqual(textured_mesh.num_vertices(), mesh.num_vertices())
        self.assertEqual(textured_mesh.num_indices(), mesh.num_indices())
        self.assertEqual(textured_mesh.normals().shape, mesh.vertices().shape)
        self.assertEqual(textured_mesh.uv_coords().shape[1], mesh.num_vertices())

    def test_has_same_material(self):
        """Test for has same material properties."""
        self.assertTrue(self.mesh.has_same_material(self.mesh))

    def test_replace_geometry(self):
        """Test for replacing mesh geometry."""
        mesh = Box3d([0, 0, 0], [1, 1, 1]).to_textured_mesh()
        mesh2 = Box3d([0, 0, 0], [1, 1, 1]).to_textured_mesh()
        mesh2.merge(mesh)
        mesh.replace_geometry(mesh2)
        self.assertEqual(mesh.num_vertices(), mesh2.num_vertices())
        self.assertEqual(mesh.num_indices(), mesh2.num_indices())
        self.assertEqual(mesh.normals().shape, mesh2.vertices().shape)
        self.assertEqual(mesh.uv_coords().shape[1], mesh2.num_vertices())


if __name__ == "__main__":
    unittest.main()
