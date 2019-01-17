"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Pose3: Rigid 3D transform modeled after GTSAM Pose3.
"""

cimport numpy as np
import json
import numpy as np
import re
import xml.etree.cElementTree as ET

from sumo.base.vector import Vector3
from sumo.geometry.quaternion import Quaternion
from sumo.geometry.rot3 import Rot3



class Pose3:
    # TODO: (dellaert) T21391110 cdef np.ndarray R_, t_

    def __init__(self, R=np.identity(3, dtype=float),
                       np.ndarray t=Vector3(0, 0, 0)):
        """Create a Pose3 instance."""
        if isinstance(R, Rot3):
            self.R = R
        else:
            self.R = Rot3(R)
        self.t = t

    def __str__(self):
        return "Pose3: t={}; R=\n{}".format(str(self.t), str(self.R))

    def rotation(self):
        return self.R

    def translation(self):
        return self.t

    def compose(self, other):
        return Pose3(self.R * other.R, self.R * other.t + self.t)

    def inverse(self):
        Rt = self.R.inverse()
        return Pose3(Rt, - (Rt * self.t))

    def transform_from(self, np.ndarray p):
        """Transform point from coordinate frame."""
        return self.R * p + self.t

    def transform_all_from(self, np.ndarray p):
        """Version of transform_to that supports 2-D arrays."""
        assert p.shape[0] == 3
        return np.transpose(np.transpose(self.R * p) + self.t)

    def __mul__(self, other):
        """Overload * operator to do either compose or transform_from."""
        if isinstance(other, Pose3):
            return self.compose(other)
        else:
            assert other.shape[0] == 3, 'points should be 3*n'
            assert other.ndim <= 2, 'points have shape {}'.format(other.shape)
            if other.ndim==1:
                return self.transform_from(other)
            else:
                return self.transform_all_from(other)

    def transform_to(self, np.ndarray q):
        """Transform point to coordinate frame."""
        return self.R.unrotate(q - self.t)

    def transform_all_to(self, np.ndarray q):
        """Version of transform_to that supports 2-D arrays."""
        assert q.shape[0] == 3
        return self.R.unrotate(np.transpose(np.transpose(q) - self.t))

    def matrix34(self):
        return np.hstack((self.R.matrix(), np.reshape(self.t, (3, 1))))

    def matrix(self):
        return np.vstack((self.matrix34(), [0, 0, 0, 1]))

    def almost_equal(self, other, *args, **kwargs):
        return np.allclose(self.matrix34(), other.matrix34(), *args, **kwargs)

    def assert_almost_equal(self, other, *args, **kwargs):
        np.testing.assert_array_almost_equal(self.matrix34(), other.matrix34(), *args, **kwargs)

    def assert_equal(self, other):
        np.testing.assert_array_equal(self.matrix34(), other.matrix34())

    @staticmethod
    def draw_poses(poses, ax, length=0.01):
        """Draw poses in 3D matplotlib axes using quiver."""
        for c,color in zip(range(3),'rgb'):
            list = [np.hstack([gTc.t[:], gTc.R.matrix()[:,c]]) for gTc in poses]
            X, Y, Z, U, V, W = zip(*list)
            ax.quiver(X, Y, Z, U, V, W, length=length, color=color, pivot='tail')

    @classmethod
    def FromMatrix34(cls, T):
        if T.shape != (3,4):
            raise ValueError("Pose3.FromMatrix34 expects 3*4 matrix")
        return cls(Rot3(T[:3,:3]),T[:3,3])

    @classmethod
    def ENU_camera(cls, float roll=0, float pitch=0, float yaw=0,
                   np.ndarray position=Vector3(0, 0, 0)):
        """Create ENU_T_camera where default is upright, level, facing north.
           The transform is from camera coordinates (Z depth, Y down) to a local
           ENU frame, i.e., X=East, Y=North, Z=Up. Optionally, perturb via:
            roll  = around Z-axis, positive is fly right
            pitch = around X-axis, positive is up
            yaw   = around Y-axis, positive is fly right
            position = 3-vector in ENU frame
        """
        return cls(R=Rot3.ENU_camera(roll, pitch, yaw), t=position)

    @classmethod
    def Maya_camera(cls, float roll=0, float pitch=0, float yaw=0,
                   np.ndarray position=Vector3(0, 0, 0)):
        """Create Maya_T_camera where default is upright, level, facing -z.
           The transform is from camera coordinates (Z depth, Y down) to the
           Maya frame, i.e., right-handed coordinate frame with Y=up.
           Optionally, perturb via:
            roll  = around Z-axis, positive is fly right in radians
            pitch = around X-axis, positive is up in radians
            yaw   = around Y-axis, positive is fly right in radians
            position = 3-vector in ENU frame
        """
        return cls(R=Rot3.Maya_camera(roll, pitch, yaw), t=position)

    @classmethod
    def from_surreal(cls, json_dict):
        """ Create Pose3 from json-style dictionary.  The format is:
              {
                "QuaternionXYZW": [0,0,0,1],
                "Translation": [0,0,0]
              }
            Return:
              Pose3 instance
        """
        q_vec = np.array(json_dict['QuaternionXYZW'])
        q_vec = np.roll(q_vec, 1)  # it has to be W, X, Y, Z
        t_vec = np.array(json_dict['Translation'])
        q = Quaternion(q_vec)
        rot = q.to_rotation_matrix()
        return Pose3(rot, t_vec)

    def to_surreal(self):
        """ Convert Pose3 to surreal-style json.  See above for format.
            Returns:
              dictionary that can be 'json.dumped'
        """
        q = Quaternion(self.rotation().matrix())
        q_vec = q.as_vector()
        q_vec = np.roll(q_vec, -1)  # it has to be X, Y, Z, W
        return {'QuaternionXYZW':list(q_vec), 'Translation':list(self.t)}

    @classmethod
    def from_xml(cls, node):
        """
        Create Pose3 from xml tree (ElementTree).  The format is:
        <pose>
        <!-- Camera pose. World from Camera point transfer. 3x4 matrix, in the RDF frame convention defined above -->
        <T_wc> [ 1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0 ] </T_wc>
        </pose>

        Return:
          Pose3 instance

        Exceptions:
          RuntimeError - if xml is not in expected format
        """

        if (node.tag != 'pose'):
            raise RuntimeError("Expected 'pose' tag but got {}".format(node.tag))

        translation = rotation = None

        # Parse the child tags
        for elem in node:
            if (elem.tag == 'translation'):
                translation = np.fromstring(elem.text, sep=',')
                if (translation.size != 3):
                    raise RuntimeError("Error parsing translation. Extracted {} numbers but was expecting 3".format(translation.size))
            elif (elem.tag == 'rotation'):
                rotation = Rot3.from_xml(elem)
            else:
                raise RuntimeError("Unexpected tag {}".format(elem.tag))

        # Make sure all tags were found
        if ((translation is None) or (rotation is None)):
            raise RuntimeError("Pose tag missing required child element.")

        return Pose3(R = rotation, t=translation)


    def to_xml(self):
        """
        Convert Pose3 to xml (see above for format).

        Return:
          Element containing pose tag
        """

        pose_elem = ET.Element('pose')

        tr_elem = ET.SubElement(pose_elem, 'translation')
        # regex to remove []
        tr_elem.text = re.sub('[\[\]]', '', np.array2string(self.t, separator=', '))

        rot_elem = self.R.to_xml()
        pose_elem.append(rot_elem)

        return (pose_elem)

    @classmethod
    def from_json(cls, json_dict):
        """ Create Pose3 from json-style dictionary <json_dict>.
            The format is:
              {'Rotation':<Rot3 json>, 'Translation':[1,2,3]}
            where <Rot3 json> is the json dict for Rot3.

            Returns:
              Pose3 instance

            Exceptions:
              ValueError - if json dictionary is not in expected format
        """
        return cls(R=Rot3.from_json(json_dict['Rotation']),
                   t=Vector3(*json_dict['Translation']))

    def to_json(self):
        """ Convert Pose3 to json.  See above for format.
            Returns:
              dictionary that can be 'json.dumped'
        """
        return {'Rotation':self.R.to_json(), 'Translation':list(self.t)}
