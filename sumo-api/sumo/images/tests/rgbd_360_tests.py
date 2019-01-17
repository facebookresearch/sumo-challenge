#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Rgbd360 unit tests.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
from sumo.images.rgbd_360 import Rgbd360


class TestRgbd360(unittest.TestCase):
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
        rgbd_360 = Rgbd360(rgb, range)
        self.assertTrue(isinstance(rgbd_360, Rgbd360))

    def test_rgbd_tiff(self):
        """We can write and read a tiff file"""
        tiff_path = os.path.join(self.temp_directory, "test.tif")
        rgb = np.random.randint(0, 255, size=(5, 5, 3)).astype(np.uint8)
        # Note near plane is 0.3, which is minimum representable range
        range = np.random.uniform(0.3, 10.0, size=(100, 100)).astype(np.float32)
        rgbd_360 = Rgbd360(rgb, range)
        rgbd_360.save(tiff_path)
        rgbd_360_read = Rgbd360.load(tiff_path)
        np.testing.assert_array_equal(rgbd_360_read.rgb, rgbd_360.rgb)
        np.testing.assert_array_almost_equal(
            rgbd_360_read.range, rgbd_360.range, decimal=2
        )


if __name__ == "__main__":
    unittest.main()
