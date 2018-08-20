#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import math
import numpy as np
import unittest

from sumo.base.vector import Vector2, Vector2f, Vector3, Vector3f, on_left, unitize


class TestVector(unittest.TestCase):

    def test_Vector2(self):
        vector = Vector2(1, 2)
        self.assertEqual(vector.shape, (2,))
        self.assertEqual(vector.dtype, np.float64)

    def test_Vector2f(self):
        vector = Vector2f(1, 2)
        self.assertEqual(vector.shape, (2,))
        self.assertEqual(vector.dtype, np.float32)

    def test_Vector3(self):
        vector = Vector3(1, 2, 3)
        self.assertEqual(vector.shape, (3,))
        self.assertEqual(vector.dtype, np.float64)

    def test_Vector3f(self):
        vector = Vector3f(1, 2, 3)
        self.assertEqual(vector.shape, (3,))
        self.assertEqual(vector.dtype, np.float32)

    def test_on_left(self):
        N = Vector3(0, 0, 1)
        p = Vector3(0, 2, 0)
        a = Vector3(0, 0, 0)
        b = Vector3(1, 0, 0)
        self.assertTrue(on_left(N, p, a, b))

    def test_unitize(self):
        v = unitize(Vector3(2, 2, 2))
        expected_v = Vector3(1, 1, 1) * (math.sqrt(3) / 3)
        np.testing.assert_array_almost_equal(v, expected_v)

        v = unitize(Vector3(100, 201, 50))
        self.assertEqual(np.linalg.norm(v), 1)

        v = unitize(Vector3(0, 0, 0))
        np.testing.assert_array_almost_equal(v, Vector3(0, 0, 0))


if __name__ == "__main__":
    unittest.main()
