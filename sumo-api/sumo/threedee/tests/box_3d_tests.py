#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


Unit tests for Box3d class.
"""

import numpy as np
import unittest

from sumo.base.vector import Vector3, Vector3f
from sumo.threedee.box_3d import Box3d
from sumo.threedee.mesh import Mesh
from sumo.threedee.textured_mesh import TexturedMesh


class TestBox3d(unittest.TestCase):
    def test_unit_cube(self):
        """ Make two points inside a unit cube and verify the bounding box's
            corners are correct
        """
        corner1 = Vector3(0, 0, 0)
        corner2 = Vector3(1, 1, 1)
        box = Box3d(corner1, corner2)
        corners = box.corners()
        corners_target = np.column_stack([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]])
        np.testing.assert_array_almost_equal(corners, corners_target)
        np.testing.assert_array_almost_equal(box.min_corner, corner1)
        np.testing.assert_array_almost_equal(box.max_corner, corner2)

    def test_random_points(self):
        """ Make two random points and verify the bounding box's corners
            are correct
        """
        corner1 = Vector3(-1, -2, -3)
        corner2 = Vector3(1, 2, 3)
        box = Box3d(corner1, corner2)
        corners = box.corners()
        corners_target = np.column_stack([
            [-1.0, -2.0, 3.0],
            [1.0, -2.0, 3.0],
            [1.0, 2.0, 3.0],
            [-1.0, 2.0, 3.0],
            [-1.0, -2.0, -3.0],
            [1.0, -2.0, -3.0],
            [1.0, 2.0, -3.0],
            [-1.0, 2.0, -3.0]])
        np.testing.assert_array_equal(corners, corners_target)

    def test_equal(self):
        """
        Test == operator.
        """

        box1 = Box3d(Vector3(1, 2, 3), Vector3(0, 4, 2))
        box2 = Box3d(Vector3(1, 2, 3), Vector3(0, 4, 2))
        box3 = Box3d(Vector3(1, 2.1, 3), Vector3(0, 4, 2))

        assert(box1 == box2)
        assert(box2 != box3)

    def test_almost_equal(self):
        """
        Test almost_equal function.
        """

        box1 = Box3d(Vector3(1.000000001, 2.000000001, 3.000001),
                     Vector3(-0.00000001, 3.99999999, 2.000001))
        box2 = Box3d(Vector3(1.0, 2, 3), Vector3(0, 4, 2))

        box3 = Box3d(Vector3(-1, -1, -1), Vector3(1, 1, 1))

        self.assertTrue(box1.almost_equal(box2, atol=0.01))
        self.assertFalse(box1.almost_equal(box2, atol=0.000000000001))
        self.assertFalse(box1.almost_equal(box3))

    def test_xml(self):
        """
        Test conversion to and from xml.  This is just a basic functionality
        test of a round trip from Box3d to xml and back.
        """
        box = Box3d(corner1=Vector3(1.5, 2.6, 3.7),
                    corner2=Vector3(5.1, 6.2, 7.9))
        box_xml = box.to_xml()
        assert(box_xml.tag == 'box3d')
        box_rt = Box3d.from_xml(box_xml)
        assert(box.almost_equal(box_rt))

    def _check_mesh_helper(self, mesh):
        """
        Checks structure of Mesh or TexturedMesh
        """
        # Check indices
        indices = mesh.indices()
        self.assertEqual(indices.shape, (6 * 6, ))
        np.testing.assert_array_equal(indices[:6], [0, 1, 2, 0, 2, 3])
        np.testing.assert_array_equal(indices[-6:], [20, 21, 22, 20, 22, 23])

        # Check 3D mesh
        vertices = mesh.vertices()
        self.assertEqual(vertices.shape, (3, 4 * 6))
        # Check BACK, where y = -1
        np.testing.assert_array_equal(
            vertices[:, :4],
            np.column_stack(
                [
                    Vector3f(0, 0, 1),
                    Vector3f(1, 0, 1),
                    Vector3f(1, 1, 1),
                    Vector3f(0, 1, 1)
                ]
            )
        )
        # Check right, back, left up
        np.testing.assert_array_equal(vertices[0, 4:8], (1, 1, 1, 1))
        np.testing.assert_array_equal(vertices[2, 8:12], (0, 0, 0, 0))
        np.testing.assert_array_equal(vertices[0, 12:16], (0, 0, 0, 0))
        np.testing.assert_array_equal(vertices[1, 16:20], (1, 1, 1, 1))
        # Check bottom
        np.testing.assert_array_equal(
            vertices[:, -4:],
            np.column_stack(
                [
                    Vector3f(0, 0, 0),
                    Vector3f(1, 0, 0),
                    Vector3f(1, 0, 1),
                    Vector3f(0, 0, 1)
                ]
            )
        )

        # check cpp_normals
        normals = mesh.normals()
        # normals for front, right, back, left, top, bot
        dir = [
            (0, 0, 1), (1, 0, 0), (0, 0, -1), (-1, 0, 0), (0, 1, 0), (0, -1, 0)
        ]
        for j in range(6):
            np.testing.assert_array_equal(normals[:, j * 4], dir[j])

    def test_mesh(self):
        box = Box3d(corner1=Vector3(0, 0, 0), corner2=Vector3(1, 1, 1))
        mesh = box.to_mesh()
        self.assertIsInstance(mesh, Mesh)
        self._check_mesh_helper(mesh)

        tex_mesh = box.to_textured_mesh(color=np.array([71, 57, 8], dtype=np.uint8))
        self.assertIsInstance(tex_mesh, TexturedMesh)
        self._check_mesh_helper(tex_mesh)

    def test_volume(self):
        """Test computation of box volume"""
        box = Box3d(corner1=Vector3f(2, 4, 6), corner2=Vector3f(0, 2, 3))
        self.assertEqual(box.volume(), 12)

    def test_center(self):
        """
        Test center function
        """
        box = Box3d(corner1=Vector3f(2, 4, 6), corner2=Vector3f(0, 2, 3))
        expected_center = Vector3f(1, 3, 4.5)
        np.testing.assert_array_almost_equal(box.center(), expected_center)

    def test_merge(self):
        box1 = Box3d(corner1=Vector3f(0, 1, 0), corner2=Vector3f(1, 2, 1))
        box2 = Box3d(corner1=Vector3f(1, 0, 0), corner2=Vector3f(2, 1, 2))
        merged = Box3d.merge([box1, box2])
        np.testing.assert_array_equal(merged.min_corner, Vector3(0, 0, 0))
        np.testing.assert_array_equal(merged.max_corner, Vector3(2, 2, 2))


if __name__ == '__main__':
    unittest.main()
