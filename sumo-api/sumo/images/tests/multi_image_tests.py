#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Multi Image Tiff unit tests.
"""

import numpy as np
import os
import shutil
import tempfile
import unittest

from libfb.py import parutil

from sumo.geometry.inverse_depth import depth_image_of_inverse_depth_map
from sumo.images.multi_image_tiff import MultiImageTiff, MultiImagePageType

PATH = parutil.get_file_path('sumo/images/test_data')


class TestMultiImageTiff(unittest.TestCase):
    def setUp(self):
        """Create temporary outout directory."""
        self.temp_directory = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up: remove temporary outout directory."""
        shutil.rmtree(self.temp_directory)

    def test_constructor(self):
        w, h = 360, 180
        rgb = np.zeros((h, w, 3), np.uint8)
        category = np.zeros((h, w), np.uint16)
        instance = np.zeros((h, w), np.uint16)
        range = np.zeros((int(h / 2), int(w / 2)), np.float32)

        pages = {
            MultiImagePageType.RGB: rgb,
            MultiImagePageType.Depth: range,
            MultiImagePageType.Category: category,
            MultiImagePageType.Instance: instance
        }

        multi = MultiImageTiff(pages)
        self.assertEqual(multi.rgb.shape, (h, w, 3))
        self.assertEqual(multi.range.shape, (h / 2, w / 2))
        self.assertEqual(multi.category.shape, (h, w))
        self.assertEqual(multi.instance.shape, (h, w))

    def test_tiff_read(self):
        tiff_path = os.path.join(PATH, 'checkered360-rgbd.tif')
        multi = MultiImageTiff.load(tiff_path)
        w, h = 360, 180
        self.assertEqual(multi.rgb.shape, (h, w, 3))
        self.assertEqual(multi.range.shape, (h / 2, w / 2))
        # Check that range in middle of image is indeed ~ 5 m.
        # agreeing with generated image from 360rgbd01_generate.ipynb
        self.assertAlmostEqual(
            multi.range[int(h / 4), int(w / 4)], 5.0, places=0
        )

    def test_tiff_read_failure(self):
        with self.assertRaises(RuntimeError):
            MultiImageTiff.load("dummy")

    def test_tiff_io(self):
        """Round-trip TIFF test that checks both dimensions and contents."""
        w, h = 1000, 600
        rgb = np.zeros((h, w, 3), np.uint8)
        range = np.full((int(h / 2), int(w / 2)), 2.0, np.float32)
        multi = MultiImageTiff({
            MultiImagePageType.RGB: rgb,
            MultiImagePageType.Depth: range
        })

        tiff_path = os.path.join(self.temp_directory, 'test-multi_image_tests1.tif')
        multi.save(tiff_path, near=1.0)
        read_near1 = MultiImageTiff.load(tiff_path)

        tiff_path2 = os.path.join(self.temp_directory, 'test-multi_image_tests2.tif')
        multi.save(tiff_path2, near=2.0)
        read_near2 = MultiImageTiff.load(tiff_path2)

        self.assertEqual(read_near1.rgb.shape, (h, w, 3))
        self.assertEqual(read_near1.range.shape, (h / 2, w / 2))
        np.testing.assert_array_equal(read_near1.rgb, multi.rgb)
        np.testing.assert_array_almost_equal(
            read_near1.range, multi.range, decimal=4)
        np.testing.assert_array_almost_equal(
            read_near1.range, read_near2.range, decimal=4)

    def test_inverse_range(self):
        w, h = 100, 60
        rgb = np.zeros((h, w, 3), np.uint8)
        inverse_range = np.full((h, w), 200, np.uint16)
        range = depth_image_of_inverse_depth_map(inverse_range)
        page_map = {
            MultiImagePageType.RGB: rgb,
            MultiImagePageType.InverseDepth: inverse_range
        }
        multi = MultiImageTiff(page_map)

        tiff_path = os.path.join(self.temp_directory, 'test-invdepth.tiff')
        multi.save(tiff_path)
        read_inverse_range = MultiImageTiff.load(tiff_path)
        np.testing.assert_array_almost_equal(read_inverse_range.range, range)


if __name__ == '__main__':
    unittest.main()
