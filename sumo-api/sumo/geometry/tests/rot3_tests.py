#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


Rot3 unit tests.
"""

import unittest
import numpy as np
import xml.etree.cElementTree as ET

from sumo.base.vector import Vector3
from sumo.geometry.rot3 import Rot3, ENU_R_CAMERA

UNIT_X = Vector3(1, 0, 0)
UNIT_Y = Vector3(0, 1, 0)
UNIT_Z = Vector3(0, 0, 1)


class TestRot3(unittest.TestCase):
    def setUp(self):
        # Camera with z-axis looking at world y-axis
        wRc = np.transpose(
            np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        )
        self.rot = Rot3(wRc)

    def test_RollPitchYaw(self):
        roll, pitch, yaw = 0.1, 0.2, 0.3
        np.testing.assert_array_equal(
            Rot3.Rx(roll).matrix(), Rot3.AxisAngle(UNIT_X, roll).matrix()
        )
        np.testing.assert_array_equal(
            Rot3.Ry(pitch).matrix(), Rot3.AxisAngle(UNIT_Y, pitch).matrix()
        )
        np.testing.assert_array_equal(
            Rot3.Rz(yaw).matrix(), Rot3.AxisAngle(UNIT_Z, yaw).matrix()
        )

    def test_matrix(self):
        expected = np.matrix('1, 0, 0;0, 0, 1;0, -1, 0')
        np.testing.assert_array_equal(self.rot.matrix(), expected)

    def test_rotate(self):
        cP = Vector3(0, 0, 5)
        wP = self.rot.rotate(cP)
        np.testing.assert_array_equal(wP, Vector3(0, 5, 0))
        np.testing.assert_array_equal(self.rot * cP, Vector3(0, 5, 0))
        np.testing.assert_array_equal(self.rot.unrotate(wP), cP)

    def test_ENU_camera(self):
        d = 0.1  # positive pitch
        wRt = Rot3.ENU_camera(pitch=d)
        expected = np.transpose(np.array([[1, 0, 0], [0, d, -1], [0, 1, d]]))
        np.testing.assert_array_almost_equal(wRt.matrix(), expected, decimal=2)

    def test_ENU_camera_w_photosphere(self):
        # Make sure we agree with conventions at
        # https://developers.google.com/streetview/spherical-metadata
        pitch = 0.1
        roll = 0.1
        wRt = Rot3.ENU_camera(pitch=pitch, roll=roll)

        expected_wRr = Rot3.Rx(pitch) * Rot3.Ry(roll)  # from rotated to ENU
        # But our tilted frame uses the camera convention.
        # TODO: maybe it should not! Why do we do this?
        expected_wRt = Rot3(expected_wRr * ENU_R_CAMERA)
        np.testing.assert_array_almost_equal(
            wRt.matrix(), expected_wRt.matrix(), decimal=2
        )

    def test_to_xml(self):
        """
        Test conversion to xml.
        """
        rot = Rot3(R=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        rot_xml = rot.to_xml()
        expected_xml = '<rotation><c1>1, 4, 7</c1><c2>2, 5, 8</c2><c3>3, 6, 9</c3>\
</rotation>'

        self.assertEqual(ET.tostring(rot_xml, encoding='unicode'), expected_xml)

    def test_from_xml(self):
        """Conversion from xml"""

        # test common case
        s = '<rotation><c1>1, 4, 7</c1><c2>2, 5, 8</c2><c3>3, 6, 9</c3></rotation>'
        rot_xml = ET.fromstring(s)
        rot = Rot3.from_xml(rot_xml)
        expected_rot = Rot3(R=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        rot.assert_almost_equal(expected_rot)

        # test missing field
        s = '<rotation><c1>1, 4, 7</c1><c3>3, 6, 9</c3></rotation>'
        self.assertRaises(ValueError, Rot3.from_xml, ET.fromstring(s))

        # test incorrect field
        s = '<rotation><foobar>1, 4, 7</foobar><c2>2, 5, 8</c2><c3>3, 6, 9</c3>\
</rotation>'

        self.assertRaises(ValueError, Rot3.from_xml, ET.fromstring(s))

    def test_json(self):
        """Test conversion to/from json dict."""
        json_dict = self.rot.to_json()
        decoded_rot = Rot3.from_json(json_dict)
        decoded_rot.assert_almost_equal(self.rot)


if __name__ == '__main__':
    unittest.main()
