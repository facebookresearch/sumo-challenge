#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

ColoredCategory class tests.
"""

import numpy as np
import unittest

from libfb.py import parutil

from sumo.base.colored_category import ColoredCategory

CSV_PATH = parutil.get_file_path(
    'sumo/base/test_data/categories.csv')


class TestColoredCategory(unittest.TestCase):
    def setUp(self):
        self.colored_categoty = ColoredCategory(CSV_PATH)

    def test_conversion(self):
        """Tests creation of object"""
        self.assertEqual(
            self.colored_categoty.category_id_to_rgb(133), (205, 145, 158)
        )
        self.assertEqual(
            self.colored_categoty.category_name_to_rgb('shoes'), (205, 145, 158)
        )

    def test_lut(self):
        """Tests LUT property."""
        lut = self.colored_categoty.LUT
        self.assertIsInstance(lut, np.ndarray)
        self.assertGreater(lut.shape[0], 0)
        self.assertGreater(lut.shape[1], 0)
        np.testing.assert_array_equal(lut[1], np.array([30, 144, 255]))

    def test_image_conversion(self):
        """Tests conversion of index to rgb based on category mapping."""
        indexed_im = np.array([[1, 2], [3, 4]]).astype(np.uint16)
        color_im = self.colored_categoty.convert_to_rgb_im(indexed_im)
        gt_color_im = np.zeros((2, 2, 3), dtype=np.uint8)
        gt_color_im[0, 0] = np.array([30, 144, 255])
        gt_color_im[0, 1] = np.array([122, 197, 205])
        gt_color_im[1, 0] = np.array([110, 123, 139])
        gt_color_im[1, 1] = np.array([0, 178, 238])
        np.testing.assert_array_equal(color_im, gt_color_im)


if __name__ == '__main__':
    unittest.main()
