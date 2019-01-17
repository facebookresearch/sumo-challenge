#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


VoxelGrid class unit tests.
"""

import numpy as np
import os
import shutil  # for rmtree
import tempfile
import unittest

from sumo.threedee.box_3d import Box3d
from sumo.threedee.voxel_grid import VoxelGrid, _linearize_voxels, _vectorize_voxels
from sumo.base.vector import Vector3f


class TestVoxelGrid(unittest.TestCase):
    def test_constructor(self):
        """
        Creates a simple voxel grid and checks its attributes.
        """
        expected_min_corner = Vector3f(-5.1, -6.2, -7.5)
        vg = VoxelGrid(0.2, expected_min_corner)
        self.assertAlmostEqual(vg.voxel_size, 0.2)
        np.testing.assert_array_equal(vg.min_corner, expected_min_corner)

    def test_conversions(self):
        """
        Tests _linearize_voxels and _vectorize_voxels by doing a round trip.
        Also check mid-point result.
        """
        voxels = np.array([[0, 0, 0], [1, 2, 2], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        size_x, size_y = 2, 3
        expected_indices = np.array([0, 17, 1, 2, 6])
        indices = _linearize_voxels(voxels, size_x, size_y)
        np.testing.assert_array_equal(indices, expected_indices)
        round_trip_voxels = _vectorize_voxels(indices, size_x, size_y)
        np.testing.assert_array_equal(round_trip_voxels, voxels)

    def test_constructor_with_points(self):
        """
        Tests voxel_centers.  Verifies that multiple points in the same voxel
        are deduplicated.
        """
        points = np.array([
            [0.1, 0.1, 0.1],
            [1.1, 1.1, 1.1],
            [1.3, 1.2, 1.4]])
        vg = VoxelGrid(0.5, min_corner=Vector3f(0, 0, 0), points=points)
        centers = vg.voxel_centers()
        expected_centers = np.array([
            [0.25, 0.25, 0.25],
            [1.25, 1.25, 1.25]])
        np.testing.assert_array_almost_equal(centers, expected_centers)

    def test_min_corner_offset(self):
        """
        Test with a non-zero min_corner.
        """
        points = np.array([
            [0.1, 0.1, 0.1],
            [1.1, 2.1, 3.1],
            [1.3, 2.2, 3.4]])
        vg = VoxelGrid(1, min_corner=Vector3f(-1, -2, -3), points=points)
        centers = vg.voxel_centers()
        expected_centers = np.array([
            [0.5, 0.5, 0.5],
            [1.5, 2.5, 3.5]])
        np.testing.assert_array_almost_equal(centers, expected_centers)

    def test_min_corner_check(self):
        """
        Test adding point less than min_corner.
        """
        points = np.array([
            [0.1, 0.1, 0.1],
            [1.1, 2.1, 3.1],
            [1.3, 2.2, 3.4],
            [0, 0, -3.1]])
        self.assertRaises(ValueError, VoxelGrid, voxel_size=1,
            min_corner=Vector3f(-1, -2, -3), points=points)

    def test_add_points(self):
        """
        Add additional points to an initial set.  Checks that redundant
        voxels are counted only once.
        """

        points = np.array([
            [0.1, 0.1, 0.1],
            [1.1, 2.1, 3.1],
            [1.3, 2.2, 3.4]])
        vg = VoxelGrid(1, min_corner=Vector3f(-1, -2, -3), points=points)
        points2 = np.array([
            [0.2, 0.2, 0.3],
            [2, 2, 2],
            [4, 4, 4]])
        vg.add_points(points2)
        centers = vg.voxel_centers()
        expected_centers = np.array([
            [0.5, 0.5, 0.5],
            [2.5, 2.5, 2.5],
            [1.5, 2.5, 3.5],
            [4.5, 4.5, 4.5]])
        np.testing.assert_array_almost_equal(centers, expected_centers)

    def test_file_io(self):
        """
        Test round trip writing and reading a simple h5 file.
        """
        temp_directory = tempfile.mkdtemp()
        filename = os.path.join(temp_directory, "test.h5")

        points = np.array([
            [0.1, 0.1, 0.1],
            [1.1, 2.1, 3.1],
            [1.3, 2.2, 3.4]])
        voxel_size = 0.5
        min_corner = Vector3f(-1, -2, -3)
        vg = VoxelGrid(voxel_size, min_corner=min_corner, points=points)

        # test writing
        vg.save(filename)
        self.assertTrue(os.path.isfile(filename))

        # test reading
        vg2 = VoxelGrid.from_file(filename)
        self.assertAlmostEquals(voxel_size, vg2.voxel_size)
        np.testing.assert_array_almost_equal(vg.min_corner, vg2.min_corner)
        shutil.rmtree(temp_directory)

    def test_bounds(self):
        points = np.array([
            [0.6, 0.1, -0.1],
            [1.1, 2.1, 3.1],
            [1.3, 2.2, 3.4]])
        min_corner = Vector3f(-1, -2, -3)
        vg = VoxelGrid(0.5, min_corner=min_corner, points=points)
        bounds = vg.bounds()
        expected_bounds = Box3d(Vector3f(0.5, 0, -0.5), Vector3f(1.5, 2.5, 3.5))
        self.assertAlmostEquals(bounds, expected_bounds)
