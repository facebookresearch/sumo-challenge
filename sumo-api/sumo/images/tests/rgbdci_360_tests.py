#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Rgbd360 unit tests.
"""

import numpy as np
import os
import shutil
import tempfile
import unittest

from sumo.images.rgbdci_360 import Rgbdci360


class TestRgbdci360(unittest.TestCase):
    def setUp(self):
        """Create temporary outout directory."""
        self.temp_directory = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up: remove temporary outout directory."""
        shutil.rmtree(self.temp_directory)

    def test_constructor(self):
        """ Test that we can make an instance """
        rgb = np.zeros((10, 10, 3), np.uint8)
        range = np.zeros((10, 10), np.float32)
        category = np.zeros((10, 10), np.uint16)
        instance = np.zeros((10, 10), np.uint16)
        rgbdci_360 = Rgbdci360(rgb, range, category, instance)
        self.assertTrue(isinstance(rgbdci_360, Rgbdci360))

    def test_rgbdci_tiff(self):
        """We can write and read a tiff file"""
        tiff_path = os.path.join(self.temp_directory, "test.tif")
        rgb = np.random.randint(0, 255, size=(5, 5, 3)).astype(np.uint8)
        # Note near plane is 0.3, which is minimum representable range
        range = np.random.uniform(0.3, 10.0, size=(100, 100)).astype(np.float32)
        category = np.random.randint(0, 32000, size=(100, 100)).astype(np.uint16)
        instance = np.random.randint(0, 32000, size=(100, 100)).astype(np.uint16)
        rgbdci_360 = Rgbdci360(rgb, range, category, instance)
        rgbdci_360.save(tiff_path)
        rgbdci_360_read = Rgbdci360.load(tiff_path)
        np.testing.assert_array_equal(rgbdci_360_read.rgb, rgbdci_360.rgb)
        np.testing.assert_array_almost_equal(
            rgbdci_360_read.range, rgbdci_360.range, decimal=2
        )
        np.testing.assert_array_equal(rgbdci_360_read.category, rgbdci_360.category)
        np.testing.assert_array_equal(rgbdci_360_read.instance, rgbdci_360.instance)

    def test_create_point_cloud(self):
        "Create point cloud from random data and check the number of points."

        rgb = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        # Note near plane is 0.3, which is minimum representable range
        range = np.random.uniform(0.3, 10.0, size=(100, 100)).astype(np.float32)
        category = np.random.randint(0, 32000, size=(100, 100)).astype(np.uint16)
        instance = np.random.randint(0, 32000, size=(100, 100)).astype(np.uint16)
        rgbdci_360 = Rgbdci360(rgb, range, category, instance)

        point_cloud = rgbdci_360.create_point_cloud()
        self.assertEqual(point_cloud.num_points(), 10000)


if __name__ == "__main__":
    unittest.main()
