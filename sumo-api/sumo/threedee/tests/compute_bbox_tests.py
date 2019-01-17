#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


Unit tests for ComputeBbox class.
"""

import numpy as np
import unittest

from sumo.base.vector import Vector3, Vector3f
from sumo.threedee.compute_bbox import ComputeBbox
from sumo.threedee.gltf_model import GltfModel
from sumo.threedee.mesh import Mesh


class TestComputeBbox(unittest.TestCase):
    def test_unit_cube(self):
        """ Make three points inside a unit cube and verify the bounding box's
            corners are correct
        """
        point1 = Vector3(1, 0, 0)
        point2 = Vector3(0, 1, 0)
        point3 = Vector3(0, 0, 1)
        point4 = Vector3(0.5, 0.5, 0.5)
        points = np.column_stack([point1, point2, point3, point4])
        box = ComputeBbox().from_point_cloud(points)
        corners = box.corners()
        corners_target = np.column_stack(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(corners, corners_target)

    def test_random_points(self):
        """ Make three random points and verify the bounding box's corners are
            correct
        """
        point1 = Vector3f(-1, -2, -3)
        point2 = Vector3f(0, 2, 1)
        point3 = Vector3f(1, 2, 3)
        points = np.column_stack([point1, point2, point3])
        box = ComputeBbox().from_point_cloud(points)
        corners = box.corners()
        corners_target = np.column_stack(
            [
                [-1.0, -2.0, 3.0],
                [1.0, -2.0, 3.0],
                [1.0, 2.0, 3.0],
                [-1.0, 2.0, 3.0],
                [-1.0, -2.0, -3.0],
                [1.0, -2.0, -3.0],
                [1.0, 2.0, -3.0],
                [-1.0, 2.0, -3.0],
            ]
        )
        np.testing.assert_array_equal(corners, corners_target)

    def test_from_mesh(self):
        """Make a mesh and verify the bounding box's corners are correct."""
        indices = np.array([0, 1, 2, 1, 3, 2, 2, 3, 4], dtype=np.uint32)
        vertices = np.column_stack(
            [
                Vector3f(0, 0, 0),
                Vector3f(1, 0, 0),
                Vector3f(0, 1, 0),
                Vector3f(1, 1, 0),
                Vector3f(1, 1, 1),
            ]
        )
        normals = np.column_stack([Vector3f(0, 0, 1)] * 5)
        self.assertEqual(vertices.shape, normals.shape)
        self.assertEqual(indices.shape, (9,))
        mesh = Mesh(indices, vertices, normals)

        box = ComputeBbox().from_mesh(mesh)
        corners = box.corners()
        corners_target = np.column_stack(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(corners, corners_target)

    def test_empty(self):
        """
        Test computing bbox for an empty point cloud.  This should produce a
        default bbox with corners at (0,0,0).
        """
        points = np.ndarray(shape=(3, 0))
        box = ComputeBbox().from_point_cloud(points)
        np.testing.assert_array_equal(box.min_corner, np.array([0, 0, 0]))
        np.testing.assert_array_equal(box.max_corner, np.array([0, 0, 0]))

    def test_from_gltf_model(self):
        """Verify bounding box computed from a GltfModel."""
        object = GltfModel.example()

        box = ComputeBbox().from_gltf_object(object)
        np.testing.assert_array_equal(box.min_corner, Vector3(-1, -1, -1))
        np.testing.assert_array_equal(box.max_corner, Vector3(1, 1, 1))


if __name__ == "__main__":
    unittest.main()
