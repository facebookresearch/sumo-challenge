#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


Quaternion unit tests.
"""

import math
import numpy as np
import unittest

from sumo.base.vector import Vector4
from sumo.geometry.quaternion import Quaternion
from sumo.geometry.rot3 import Rot3


class TestQuaternion(unittest.TestCase):
    def setUp(self):
        self.theta1 = 0.123
        self.theta2 = math.radians(30)

        self.q1 = Quaternion()
        # q2 is a quaternion by rotating 0.123 (radians) w.r.t x axis
        self.q2 = Quaternion(
            Vector4(math.cos(self.theta1 / 2), math.sin(self.theta1 / 2), 0, 0)
        )

        # q3 is a quaternion by rotaing 30 (degrees) w.r.t z axis
        self.q3 = Quaternion(
            Vector4(math.cos(self.theta2 / 2), 0, 0, math.sin(self.theta2 / 2))
        )

    def test_constructor(self):
        np.testing.assert_array_equal(self.q1.as_vector(), Vector4(1, 0, 0, 0))
        np.testing.assert_array_almost_equal(
            self.q2.as_vector(),
            Vector4(math.cos(self.theta1 / 2), math.sin(self.theta1 / 2), 0, 0)
        )

    # TODO: def test_quaternion_from_axis_angle(self):
    #     self.q1.quaternion_from_axis_angle([0, 0, 1], self.theta)
    #     self.assertTrue(np.allclose(self.q1.values, [0.99810947, 0.06146124, 0, 0]))

    def test_to_rotation_matrix(self):
        rot1 = self.q1.to_rotation_matrix()
        np.testing.assert_array_almost_equal(rot1, np.identity(3))

        M2 = Rot3.Rx(self.theta1).matrix()
        rot2 = self.q2.to_rotation_matrix()
        self.assertTrue(np.allclose(rot2, M2))

        M3 = Rot3.Rz(self.theta2).matrix()
        rot3 = self.q3.to_rotation_matrix()
        np.testing.assert_array_almost_equal(rot3, M3)

    def test_matrix_roundtrip(self):
        rot2 = self.q2.to_rotation_matrix()
        q = Quaternion(rot2)
        np.testing.assert_array_almost_equal(q.as_vector(), self.q2.as_vector())


if __name__ == '__main__':
    unittest.main()
