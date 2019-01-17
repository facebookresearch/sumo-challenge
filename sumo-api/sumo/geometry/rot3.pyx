"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


Rot3: Rigid 3D transform modeled after GTSAM Rot3.
"""

import math
import numpy as np
cimport numpy as np
import re
import xml.etree.cElementTree as ET

# Default camera rotation facing due north in ENU frame
# see https://developers.google.com/streetview/spherical-metadata
ENU_R_CAMERA = np.transpose(
    np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float)
)

# Default camera rotation taking Maya convention
MAYA_R_CAMERA = np.transpose(
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)
)


class Rot3:
    def __init__(self, np.ndarray R=np.identity(3, dtype=float)):
        """Create a Rot3 instance."""
        self.R = R

    def matrix(self):
        """Return 3*3 matrix."""
        return self.R

    def __call__(self, size_t j):
        """Return 3*1 column."""
        return self.R[:,j]

    def inverse(self):
        """Return R^t."""
        return Rot3(np.transpose(self.R))

    def rotate(self, np.ndarray p):
        """Return R*p."""
        return self.R.dot(p)

    def unrotate(self, np.ndarray q):
        """Return R^T*p."""
        return np.transpose(self.R).dot(q)

    def __str__(self):
        """String representation is matrix."""
        return str(self.R)

    def __mul__(self, other):
        """Overload * operator so we don't need np.dot all over."""
        if isinstance(other, Rot3):
            return Rot3(np.dot(self.R, other.R))  # returns Rot3
        else:
            # try dot, works with 3*n matrices, including rotation matrices
            return np.dot(self.R, other)  # returns np.array

    @classmethod
    def FromScalars(cls, R11, R12, R13, R21, R22, R23, R31, R32, R33):
        R = np.array(
            [[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]], dtype=np.float
        )
        return cls(R)

    @classmethod
    def FromColumns(cls, x, y, z):
        R = np.stack((x, y, z), axis=1)
        return cls(R)

    @classmethod
    def Rx(cls, t):
        st, ct = math.sin(t), math.cos(t)
        return cls.FromScalars(1, 0, 0, 0, ct, -st, 0, st, ct)

    @classmethod
    def Ry(cls, t):
        st, ct = math.sin(t), math.cos(t)
        return cls.FromScalars(ct, 0, st, 0, 1, 0, -st, 0, ct)

    @classmethod
    def Rz(cls, t):
        st, ct = math.sin(t), math.cos(t)
        return cls.FromScalars(ct, -st, 0, st, ct, 0, 0, 0, 1)

    @classmethod
    def AxisAngle(cls, np.ndarray d, double a):
        """ Create rotation matrix R = dd^T + cos(a) (I - dd^T) + sin(a) skew(d)
            Keyword arguments:
              d -- a unit vector specifying the axis of rotation
              a -- angle around the rotation axis, in radians
        """
        # inspired by from CameraGen/symmetry.py, which is not in buck yet
        eye = np.eye(3)
        ddt = np.outer(d, d)
        skew = np.array(
            [[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]],
            dtype=np.float64
        )
        return cls(ddt + np.cos(a) * (eye - ddt) + np.sin(a) * skew)

    def almost_equal(self, other, *args, **kwargs):
        return np.allclose(self.matrix(), other.matrix(), *args, **kwargs)

    def assert_almost_equal(self, other, *args, **kwargs):
        """
        Raises an AssertionError if two Rot3 instances (<self> and <other>)
        are not equal to desired precision.  See numpy.testing docs for details.
        """
        np.testing.assert_array_almost_equal(self.matrix(), other.matrix(), *args, **kwargs)

    def assert_equal(self, other):
        """
        Raises an AssertionError if two Rot3 instances (<self> and <other>)
        are not equal.
        """
        np.testing.assert_array_equal(self.matrix(), other.matrix())

    @classmethod
    def ENU_camera(cls, roll=0, pitch=0, yaw=0):
        """Create ENU_R_camera where default is upright, level, facing north.
           The rotation is from camera coordinates (Z depth, Y down) to a local
           ENU frame, i.e., X=East, Y=North, Z=Up. Optionally, perturb via:
            roll  = around Z-axis, positive is fly right
            pitch = around X-axis, positive is up
            yaw   = around Y-axis, positive is fly right
        """
        return cls(ENU_R_CAMERA) * Rot3.Ry(yaw) * Rot3.Rx(pitch) * Rot3.Rz(roll)

    @classmethod
    def Maya_camera(cls, roll=0, pitch=0, yaw=0):
        """Create maya_R_camera. The Maya frame is a right-handed coordinate
           frame where Y is up.
           The default returns a camera which is pointing in the -z direction.
           Optionally, perturb via roll/pitch/yaw in the camera frame:
            roll  = around Z-axis, positive is fly right in radians
            pitch = around X-axis, positive is up in radians
            yaw   = around Y-axis, positive is fly right in radians
        """
        return cls(MAYA_R_CAMERA) * Rot3.Ry(yaw) * Rot3.Rx(pitch) * Rot3.Rz(roll)

    @classmethod
    def from_xml(cls, base_elem):
        """
        Create Rot3 from xml tree (ElementTree Element) <base_elem>.
        The format is:
        <rotation>
          <c1> 1.0, 0, 0 </c1>
          <c2> 0, 1.0, 0 </c2>
          <c3> 0, 0, 1.0 </c3>
        </rotation>

        Return:
          Rot3 instance

        Exceptions:
          ValueError - if xml is not in expected format
        """

        if (base_elem.tag != 'rotation'):
            raise ValueError("Expected 'rotation' tag but got {}".format(base_elem.tag))

        col_lookup = {'c1': 0, 'c2': 1, 'c3': 2}
        found = {}
        R = np.zeros((3,3))
        for elem in base_elem:
            try:
                R[:,col_lookup[elem.tag]] = np.fromstring(elem.text, sep=',')
                found[elem.tag] = True
            except:
                raise ValueError("Expected c1, c2, or c3, but got {}".format(elem.tag))
        if (len(found) != 3):
            raise ValueError("rotation only has {} of 3 columns".format(len(found)))
        return cls(R = R)

    def to_xml(self):
        """
        Convert Rot3 to xml.  See above for format.

        Return:
          ElementTree Element containing pose tag
        """

        base_elem = ET.Element('rotation')

        for c in range(3):
            elem = ET.SubElement(base_elem, 'c' + str(c+1))
            elem.text =  re.sub('[\[\]]', '', np.array2string(self.R[:,c], separator=', '))

        return (base_elem)

    @classmethod
    def from_json(cls, json_dict):
        """ Create Rot3 from json-style dictionary <json_dict>.
            The format is a dict with the three *columns* of the 3*3 matrix:
              {
                'X': [1.0, 0, 0]
                'Y': [0, 1.0, 0]
                'Z': [0, 0, 1.0]
              }

            Returns:
              Rot3 instance

            Exceptions:
              ValueError - if json dictionary is not in expected format
        """
        return cls.FromColumns(json_dict['X'],json_dict['Y'],json_dict['Z'])

    def to_json(self):
        """ Convert Rot3 to json.  See above for format.
            Returns:
              dictionary that can be 'json.dumped'
        """
        R = self.matrix()
        return {field: list(R[:,i]) for i, field in enumerate(['X','Y','Z'])}
