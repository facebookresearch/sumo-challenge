#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Unit tests for inverse_depth module.
"""

import numpy as np
import unittest

from sumo.geometry.inverse_depth import (
    DEFAULT_NEAR,
    PIXEL_MAX,
    depth_image_of_inverse_depth_map,
    depth_of_inverse_depth,
    inverse_depth_map_of_depth_image,
    inverse_depth_of_depth,
    uint16_of_uint8_inverse_depth_map,
)


class TestInverseDepth(unittest.TestCase):
    def test_conversions(self):
        self.assertEqual(inverse_depth_of_depth(DEFAULT_NEAR), PIXEL_MAX)
        self.assertEqual(
            inverse_depth_of_depth(PIXEL_MAX * DEFAULT_NEAR), 1)
        self.assertEqual(
            inverse_depth_of_depth(PIXEL_MAX * 1.1), 0)

        self.assertEqual(
            inverse_depth_of_depth(0.9, near=1.0), PIXEL_MAX)
        self.assertEqual(
            inverse_depth_of_depth(1.0, near=1.0), PIXEL_MAX)
        self.assertEqual(
            inverse_depth_of_depth(2.0, near=1.0), (PIXEL_MAX + 1) / 2)
        self.assertEqual(
            inverse_depth_of_depth(PIXEL_MAX, near=1.0), 1)
        self.assertEqual(
            inverse_depth_of_depth(PIXEL_MAX * 2, near=1.0), 1)
        self.assertEqual(
            inverse_depth_of_depth(PIXEL_MAX * 2 + 1, near=1.0), 0)
        self.assertEqual(inverse_depth_of_depth(float('inf')), 0)

        self.assertEqual(
            depth_of_inverse_depth(PIXEL_MAX, near=1.0), 1.0)
        self.assertAlmostEqual(
            depth_of_inverse_depth((PIXEL_MAX + 1) / 2, near=1.0),
            1.9921875, places=1)
        self.assertEqual(
            depth_of_inverse_depth(1, near=1.0), PIXEL_MAX)
        self.assertEqual(depth_of_inverse_depth(0), float('inf'))

        self.assertEqual(
            inverse_depth_of_depth(0.1, near=0.1), PIXEL_MAX)
        self.assertEqual(
            inverse_depth_of_depth(0.2, near=0.1), int(PIXEL_MAX / 2 + 0.5))
        self.assertEqual(
            inverse_depth_of_depth(3.0, near=0.1), 2185)
        self.assertEqual(inverse_depth_of_depth(float('inf')), 0)

        self.assertEqual(
            inverse_depth_of_depth(0), 0)

        self.assertAlmostEqual(
            depth_of_inverse_depth(PIXEL_MAX, near=0.1), 0.1)
        self.assertAlmostEqual(
            depth_of_inverse_depth((PIXEL_MAX + 1) / 2, near=0.1),
            0.19921875, places=2)
        self.assertAlmostEqual(
            depth_of_inverse_depth(13, near=0.1), 504.1153846, places=5)
        self.assertEqual(
            depth_of_inverse_depth(0, near=0.1), float('inf'))

    def test_inverse_depth_map_of_depth_image(self):
        w, h = 5, 3
        depth_image = np.full((h, w), 3.0, np.float32)
        inverse_depth_map = inverse_depth_map_of_depth_image(
            depth_image, near=0.1)
        self.assertEqual(inverse_depth_map.dtype, np.uint16)
        self.assertEqual(inverse_depth_map.shape, (h, w))
        self.assertEqual(inverse_depth_map[0, 0], 2185)

    def test_depth_image_of_inverse_depth_map(self):
        w, h = 5, 3
        inverse_depth_map = np.full((h, w), 2185, np.uint16)
        depth_image = depth_image_of_inverse_depth_map(
            inverse_depth_map, near=0.1
        )
        self.assertEqual(depth_image.dtype, np.float32)
        self.assertEqual(depth_image.shape, (h, w))
        self.assertAlmostEqual(depth_image[0, 0], 3.0, places=2)

    def test_uint8_to_uint16_conversion(self):
        raw_depth = 3.
        # Get the 8-bit, 1m near plane depth representation
        inv_depth_uint8 = inverse_depth_of_depth(
            raw_depth, near=1.0)
        # Get representation using defaults
        inv_depth_uint16 = inverse_depth_of_depth(raw_depth)

        w, h = 5, 3
        image_uint8 = np.full((h, w), inv_depth_uint8, np.uint8)
        image_uint16 = np.full((h, w), inv_depth_uint16, np.uint16)

        converted_image = uint16_of_uint8_inverse_depth_map(
            image_uint8, near=1.0)

        self.assertEqual(converted_image.dtype, np.uint16)
        self.assertEqual(converted_image.shape, (h, w))
        self.assertEqual(converted_image[0, 0], image_uint16[0, 0])


if __name__ == '__main__':
    unittest.main()
