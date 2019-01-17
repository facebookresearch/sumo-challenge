#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


Mesh class unit tests.
"""

import unittest

import numpy as np
from sumo.base.vector import Vector3f
from sumo.threedee.mesh import Mesh
from sumo.threedee.box_3d import Box3d


class TestMesh(unittest.TestCase):
    def setUp(self):
        self.vertices = np.column_stack(
            [Vector3f(0, 0, 0), Vector3f(1, 0, 0), Vector3f(0, 1, 0), Vector3f(1, 1, 0)]
        )
        self.normals = np.column_stack([Vector3f(0, 0, 1)] * 4)
        self.indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)
        self.mesh = Mesh(self.indices, self.vertices, self.normals)

    def test_example(self):
        mesh = Mesh.example(2, True)
        self.assertEqual(mesh.indices().shape[0], 12 * 3)
        self.assertEqual(mesh.vertices().shape[1], 8)
        self.assertEqual(mesh.normals().shape[1], 8)

    def test_constructor(self):
        self.assertEqual(self.indices.shape, (6,))
        self.assertEqual(self.mesh.num_indices(), 6)
        self.assertEqual(self.mesh.num_vertices(), 4)
        np.testing.assert_array_equal(self.mesh.indices(), self.indices)
        np.testing.assert_array_equal(self.mesh.vertices(), self.vertices)
        np.testing.assert_array_equal(self.mesh.normals(), self.normals)

    def test_default_constructor(self):
        """Test construction of an empty mesh."""
        mesh = Mesh()
        self.assertEqual(mesh.num_vertices(), 0)
        self.assertEqual(mesh.num_indices(), 0)

    def test_cleanup(self):
        """Check removing triangles with long edges, and (0,0,0) vertices."""
        indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)
        vertices = np.column_stack(
            [
                Vector3f(0, 0, 0),
                Vector3f(1, 0, 0),
                Vector3f(0, 1, 0),
                Vector3f(100, 100, 0),
            ]
        )
        mesh = Mesh(indices, vertices, self.normals)
        mesh.cleanup_long_edges(threshold=10)
        self.assertEqual(mesh.num_indices(), 3)
        np.testing.assert_array_equal(mesh.indices(), [0, 1, 2])

        mesh.cleanup_edges_to_origin()
        self.assertEqual(mesh.num_indices(), 0)

    def test_face_normals(self):
        """Test that we can calculate face normals."""
        normals = Mesh.calculate_face_normals(self.indices, self.vertices)
        expected = np.column_stack([Vector3f(0, 0, 1), Vector3f(0, 0, 1)])
        np.testing.assert_array_equal(normals, expected)

    def test_vertex_normals(self):
        """Test that we can calculate normals."""
        normals = Mesh.estimate_normals(self.indices, self.vertices)
        expected = np.column_stack(
            [Vector3f(0, 0, 1), Vector3f(0, 0, 1), Vector3f(0, 0, 1), Vector3f(0, 0, 1)]
        )
        np.testing.assert_array_equal(normals, expected)

    def test_merge(self):
        """Test merging two meshes with common vertices."""
        vertices = np.column_stack([self.vertices, self.vertices])
        self.assertEqual(vertices.shape, (3, 8))
        normals = np.column_stack([self.normals, self.normals])

        indices1 = np.array([0, 1, 2, 4, 5, 7], dtype=np.uint32)
        mesh = Mesh(indices1, vertices, normals)

        indices2 = np.array([1, 3, 2, 4, 5, 7], dtype=np.uint32)
        mesh2 = Mesh(indices2, vertices, normals)

        mesh.merge(mesh2, 4)
        self.assertEqual(mesh.vertices().shape, (3, 12))
        expected = np.column_stack([self.vertices, self.vertices, self.vertices])
        np.testing.assert_array_equal(mesh.vertices(), expected)
        expected_indices = np.array(
            [0, 1, 2, 4, 5, 7, 1, 3, 2, 8, 9, 11], dtype=np.uint32
        )
        np.testing.assert_array_equal(mesh.indices(), expected_indices)

    def test_replace_geometry(self):
        """Test for replacing mesh geometry."""
        mesh = Box3d([0, 0, 0], [1, 1, 1]).to_mesh()
        mesh2 = Box3d([0, 0, 0], [1, 1, 1]).to_mesh()
        mesh2.merge(mesh)
        mesh.replace_geometry(mesh2)
        self.assertEqual(mesh.num_vertices(), mesh2.num_vertices())
        self.assertEqual(mesh.num_indices(), mesh2.num_indices())
        self.assertEqual(mesh.normals().shape, mesh2.vertices().shape)

    def test_has_same_material(self):
        """Check whether mesh has same material properties."""
        self.assertTrue(self.mesh.has_same_material(self.mesh))


if __name__ == "__main__":
    unittest.main()
