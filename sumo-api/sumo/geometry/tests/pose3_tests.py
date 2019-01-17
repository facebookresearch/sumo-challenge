#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Pose3 unit tests.
"""

import json
import numpy as np
import os
import unittest

from libfb.py import parutil

from sumo.base.vector import Vector3
from sumo.geometry.pose3 import Pose3

PATH = parutil.get_file_path('sumo/geometry/test_data')


class TestPose3(unittest.TestCase):
    def setUp(self):
        # Camera with z-axis looking at world y-axis
        wRc = np.transpose(
            np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        )
        t = Vector3(1, 1, 1)
        self.pose = Pose3(wRc, t)

    def test_assert_equal(self):
        self.pose.assert_equal(self.pose)
        self.pose.assert_almost_equal(self.pose, decimal=3)

    def test_ENU_camera(self):
        Pose3.ENU_camera(position=Vector3(1, 1, 1)).assert_equal(self.pose)

    def test_compose(self):
        expected = Pose3(self.pose.rotation(), Vector3(1, 5, 1))
        translate = Pose3(t=Vector3(0, 0, 4))  # translate Z axis, which is Y
        actual = self.pose.compose(translate)
        actual.assert_equal(expected)
        actual2 = self.pose * translate
        actual2.assert_equal(expected)

    def test_inverse(self):
        expected = Pose3()
        actual1 = self.pose.inverse() * self.pose
        actual1.assert_equal(expected)
        actual2 = self.pose * self.pose.inverse()
        actual2.assert_equal(expected)

    def test_transform(self):
        cP = Vector3(0, 0, 5)
        wP = self.pose.transform_from(cP)
        np.testing.assert_array_equal(wP, Vector3(1, 6, 1))
        np.testing.assert_array_equal(self.pose.transform_to(wP), cP)
        np.testing.assert_array_equal(wP, self.pose * cP)
        w_points = np.stack([wP, wP], axis=1)
        c_points = np.stack([cP, cP], axis=1)
        np.testing.assert_array_equal(
            c_points, self.pose.transform_all_to(w_points)
        )
        np.testing.assert_array_equal(
            w_points, self.pose.transform_all_from(c_points)
        )
        np.testing.assert_array_equal(w_points, self.pose * c_points)

    def test_matrix34(self):
        expected = np.matrix('1, 0, 0, 1;0, 0, 1, 1;0, -1, 0, 1')
        np.testing.assert_array_equal(self.pose.matrix34(), expected)

    def test_matrix(self):
        expected = np.matrix('1, 0, 0, 1; 0, 0, 1, 1; 0, -1, 0, 1; 0, 0, 0, 1')
        np.testing.assert_array_equal(self.pose.matrix(), expected)

    def test_xml(self):
        """
        Test conversion to and from xml.  This is just a basic functionality
        test of a round trip from Pose3 to xml and back.
        """
        pose = Pose3(
            t=Vector3(1.5, 2.6, 3.7),
            R=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        )
        pose_xml = pose.to_xml()
        self.assertEqual(pose_xml.tag, 'pose')
        pose_rt = Pose3.from_xml(pose_xml)
        pose.assert_almost_equal(pose_rt)

    def test_json(self):
        """Test conversion to/from json dict."""
        json_dict = self.pose.to_json()
        decoded_pose = Pose3.from_json(json_dict)
        decoded_pose.assert_almost_equal(self.pose)

    def test_surreal_coding(self):
        """Test conversion to/from surreal-style json dict."""
        json_dict = self.pose.to_surreal()
        decoded_pose = Pose3.from_surreal(json_dict)
        decoded_pose.assert_almost_equal(self.pose)

    def test_surreal(self):
        """Read from a Surreal json file."""

        # Expected pose.
        expected_pose = Pose3(t=Vector3(1, 2, 3))

        # Read json file
        path_to_json_linear = os.path.join(PATH, 'pose3_test.json')
        data = json.load(open(path_to_json_linear))
        pose = Pose3.from_surreal(data['T_cr'])

        # Test pose
        pose.assert_almost_equal(expected_pose)


if __name__ == '__main__':
    unittest.main()
