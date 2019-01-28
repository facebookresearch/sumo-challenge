#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import numpy as np
import os
import tempfile
import unittest

from libfb.py import parutil

from sumo.base.vector import Vector3
from sumo.threedee.point_cloud import PointCloud
from sumo.geometry.pose3 import Pose3
from sumo.geometry.rot3 import Rot3


class TestPointCloud(unittest.TestCase):
    def test_constructor(self):
        points = np.reshape(np.arange(18), (3, 6))
        cloud = PointCloud(points)
        self.assertEquals(cloud.num_points(), 6)
        np.testing.assert_array_equal(cloud.points(), points)
        self.assertFalse(cloud.colored())

    def test_colors(self):
        n = 100  # number of colors in original cloud, which we will thin to n/2
        cloud = PointCloud(np.zeros((3, n)), np.zeros((3, n), np.uint8))
        self.assertTrue(cloud.colored())

    def test_thin(self):
        n = 100  # number of points in original cloud, which we will thin to n/2
        cloud = PointCloud(np.zeros((3, n)), np.zeros((3, n), np.uint8))
        thinned = cloud.thin(int(n / 2))
        self.assertEquals(thinned.num_points(), n / 2)
        self.assertTrue(thinned.colored())
        self.assertEquals(cloud.num_points(), n)

    def test_load_ply(self):
        file_path = parutil.get_file_path(
            'sumo/threedee/test_data/example_tetrahedron.ply'
        )
        cloud = PointCloud.load_ply(file_path)
        self.assertTrue(isinstance(cloud, PointCloud))
        self.assertEquals(cloud.num_points(), 12)

    def test_write_ply(self):
        # Prepare input test signal
        folder_path = parutil.get_file_path(
            'sumo/threedee/test_data')
        file_path = os.path.join(folder_path, 'example_tetrahedron.ply')
        cloud_input = PointCloud.load_ply(file_path)

        # Write test signal into write_ply
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, 'example_write.ply')
        cloud_input.write_ply(file_path)
        cloud_output = PointCloud.load_ply(file_path)

        self.assertTrue(isinstance(cloud_output, PointCloud))
        self.assertEqual(cloud_output.num_points(), cloud_input.num_points())
        np.testing.assert_array_equal(
            cloud_output.points(), cloud_input.points()
        )
        os.remove(file_path)

    def test_write_ply_with_colors(self):
        n = 100  # number of colors in original cloud
        cloud_input = PointCloud(np.zeros((3, n)), np.zeros((3, n), np.uint8))

        # Write test signal into a file
        temp_dir = tempfile.mkdtemp()

        file_path = os.path.join(temp_dir, 'example_write.ply')
        cloud_input.write_ply(file_path)

        # TODO: The current PointCloud class does not actually read
        # colors.  Add implementation.
        # cloud_output = PointCloud.load_ply(file_path)
        # self.assertTrue(isinstance(cloud_output, PointCloud))
        # self.assertTrue(cloud_output.colored())
        # self.assertEqual(cloud_output.colors().shape, [3, n])
        os.remove(file_path)

    def test_sum(self):
        """ Test overloads of add for merging point clouds. """
        cloud = PointCloud(np.zeros((3, 10)), np.zeros((3, 10), np.uint8))
        twice = cloud + cloud
        self.assertTrue(isinstance(twice, PointCloud))
        self.assertEquals(twice.num_points(), 20)
        five = sum([twice, twice, cloud])
        self.assertTrue(isinstance(five, PointCloud))
        self.assertTrue(five.colored())
        self.assertEquals(five.num_points(), 50)
        self.assertEquals(five.colors().shape, (3, 50))

    def test_append(self):
        """ Test (imperative!) append method."""
        cloud = PointCloud(np.zeros((3, 10)), np.zeros((3, 10), np.uint8))
        cloud.append(cloud)
        self.assertEquals(cloud.num_points(), 20)
        self.assertEquals(cloud.colors().shape, (3, 20))

    def test_transform_from(self):
        """Test transform_from."""
        t = Vector3(1, 1, 1)
        R = Rot3()
        T = Pose3(R, t).inverse()
        cloud = PointCloud(np.zeros((3, 1)))
        new_cloud = cloud.transform_from(T)
        np.testing.assert_array_equal(
            new_cloud.points(), [[-1.0], [-1.0], [-1.0]]
        )

    def test_register(self):
        """ Test registering point clouds in different frames. """
        t = Vector3(1, 1, 1)
        R = Rot3()
        T = Pose3(R, t).inverse()
        cloud = PointCloud(np.zeros((3, 1)))
        registred_cloud = PointCloud.register([(Pose3(), cloud), (T, cloud)])
        np.testing.assert_array_equal(
            registred_cloud.points(),
            np.column_stack([Vector3(0, 0, 0),
                             Vector3(-1.0, -1.0, -1.0)])
        )


if __name__ == '__main__':
    unittest.main()
