"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import os
import re
import xml.etree.cElementTree as ET

from sumo.base.vector import Vector3f
from sumo.threedee.mesh import Mesh
from sumo.threedee.textured_mesh import TexturedMesh

class Box3d(object):
    """
    Axis-aligned box.
    The anchor for the box is the front-bottom-left.
    Coordinate frame is (+x right, +y up, +z toward viewer)

    Public attributes:
        min_corner (Vector3f) - Minimum corner of box. (read only)
        max_corner (Vector3f) - Maximum corner of box. (read only)
    """

    def __init__(self, corner1=Vector3f(0, 0, 0), corner2=Vector3f(0, 0, 0)):
        """
        Constructor. Initialize using two opposing corners, <corner1> and
        <corner2>.  Order does not matter.
        """

        points = np.column_stack([corner1, corner2])
        self._min_corner = np.amin(points, axis=1)
        self._max_corner = np.amax(points, axis=1)

    # note: min_corner is read-only
    @property
    def min_corner(self):
        return self._min_corner

    # note: max_corner is read-only
    @property
    def max_corner(self):
        return self._max_corner

    def point_in_box(self, point):
        """
        Checks whether the given point (np.ndarray) is inside this box
        Note: check is not inclusive of the boundary
        """
        return np.all(point > self.min_corner) and \
            np.all(point < self.max_corner)

    def corners(self):
        """
        Generate and return the corners of the box.

        Return:
        2D array with corners as column vectors.
        The order is CCW around the front and then CCW around the back (as
        viewed from the front):
            front-bot-left, front-bot-right, front-top-right, front-top-left
            back-bot-left, back-bot-right, back-top-right, back-top-left
        """
        minc = self._min_corner
        maxc = self._max_corner
        c = np.column_stack([  # corners
          [minc[0], minc[1], maxc[2]],
          [maxc[0], minc[1], maxc[2]],
          [maxc[0], maxc[1], maxc[2]],
          [minc[0], maxc[1], maxc[2]],
          [minc[0], minc[1], minc[2]],
          [maxc[0], minc[1], minc[2]],
          [maxc[0], maxc[1], minc[2]],
          [minc[0], maxc[1], minc[2]]])
        return c

    def volume(self):
        """Compute the volume of the box.  Returns volume (float)"""

        return np.prod(self._max_corner - self._min_corner)

    def center(self):
        """
        Find center of the box.

        Return:
        Vector3f - center coordinates
        """
        return (self._max_corner + self.min_corner) / 2

    def almost_equal(self, other, *args, **kwargs):
        """
        Test if two Box3d objects are equal within a tolerance.
        args and kwargs are passed through to np.isclose calls
        (see numpy docs for details).
        """
        return (np.allclose(self._min_corner, other._min_corner, *args, **kwargs) and
                np.allclose(self._max_corner, other._max_corner, *args, **kwargs))


    def __eq__(self, other, *args, **kwargs):
        """
        Test if two Box3d objects are equal.  args and kwargs are passed through to
        np.equal calls (see numpy docs for details).
        """
        return (np.array_equal(self._min_corner, other._min_corner, *args, **kwargs) and
                np.array_equal(self._max_corner, other._max_corner, *args, **kwargs))


    def __str__(self):
        """
        Print in human-readable format.
        """
        result = "min_corner: {}, max_corner: {}".format(self._min_corner, self._max_corner)
        return result


    def to_mesh(self, normals_out=True):
        """
        Convert Box3d to mesh.  The mesh has 2 faces on each side.  Corner vertices are
        duplicated to allow each face to have independent normals.

        Inputs:
        normals_out (Boolean) - If true, the normals and faces are set so they face
          outward (as if viewing a box from outside), otherwise, they are set to face inward (as if viewing the inside of a room)

        Return:
            Mesh object
        """

        vertices, indices, normals = self._to_mesh_helper(normals_out)

        return Mesh(indices, vertices, normals)

    def to_textured_mesh(self, normals_out=True,
                         color=np.full((3, 1), 255, dtype=np.uint8)):
        """
        Convert Box3d to mesh.  The mesh has 2 faces on each side.  Corner vertices are
        duplicated to allow each face to have independent normals.

        Inputs:
        normals_out (Boolean) - If true, the normals and faces are set so they face
          outward (as if viewing a box from outside), otherwise, they are set to face inward (as if viewing the inside of a room)
        color (3x1 array) - Color for base color of textured mesh.

        Return:
            TexturedMesh object
        """

        vertices, indices, normals = self._to_mesh_helper(normals_out)

        # setup texture (all texture coords are (0,0))
        uv_coords = np.zeros((2, vertices.shape[1]), dtype=np.float32)
        # convert color to repeat in 2x2 array of colors
        base_color = np.reshape(np.resize(color, (4, 3)), (2, 2, 3))

        return TexturedMesh(indices, vertices, normals, uv_coords, base_color)

    @classmethod
    def from_xml(cls, base_elem):
        """
        Create Box3d from xml tree <base_elem>.
        Format of xml is:
        <box3d> <corner1> x, y, z </corner1> <corner2> x, y, z </corner2> </box3d>

        Input:
        base_elem - cElementtree element with <box3d> tag.

        Return:
            Box3d object

        Exceptions:
            RuntimeError - if xml tree cannot be parsed.
        """

        if (base_elem.tag != 'box3d'):
            raise RuntimeError("Expected 'box3d' tag but got {}".format(base_elem.tag))

        corner1 = corner2 = None

        # Parse the child tags
        for elem in base_elem:
            if (elem.tag == 'corner1'):
                corner1 = np.fromstring(elem.text, sep=',')
                if (corner1.size != 3):
                    raise RuntimeError("Error parsing corner1. Extracted {} numbers but was expecting 3".format(corner1.size))
            elif (elem.tag == 'corner2'):
                corner2 = np.fromstring(elem.text, sep=',')
                if (corner2.size != 3):
                    raise RuntimeError("Error parsing corner2. Extracted {} numbers but was expecting 3".format(corner2.size))
            else:
                raise RuntimeError("Unexpected tag {}".format(elem.tag))

        # Make sure all tags were found
        if ((corner1 is None) or (corner2 is None)):
            raise RuntimeError("bounds tag missing required child element.")

        return cls(corner1, corner2)


    def to_xml(self):
        """
        Convert Box3d to bounds xml (see above for format).

        Return:
          Element containing <box3d> tag
        """

        base_elem = ET.Element('box3d')

        c1_elem = ET.SubElement(base_elem, 'corner1')
        c1_elem.text = re.sub('[\[\]]', '', np.array2string(self._min_corner, separator=', '))

        c2_elem = ET.SubElement(base_elem, 'corner2')
        c2_elem.text = re.sub('[\[\]]', '', np.array2string(self._max_corner, separator=', '))

        return (base_elem)

    @classmethod
    def merge(cls, list boxes):
        """ Merge boxes in the given list of Box3d instances.
            Inputs:
                boxes (list) - list of Box3d objects.
        """
        if len(boxes)==0:
            return Box3d()
        mins = np.column_stack([box.min_corner for box in boxes])
        maxs = np.column_stack([box.max_corner for box in boxes])
        min_corner = np.amin(mins, axis=1)
        max_corner = np.amax(maxs, axis=1)
        return Box3d(min_corner, max_corner)

#-------------------------------------
# End of public interface

    def _to_mesh_helper(self, normals_out):
        """
        Helper function for creating mesh from Box3d.  Creates vertices, indices,
        and normals.

        normals_out (Boolean) - If true, the normals and faces are set so they face
        outward (as if viewing a box from outside), otherwise, they are set to face
        inward (as if viewing the inside of a room)

        Return:
        tuple containing (vertices, indices, normals) arrays.
            vertices (3 x 24) np.array
            indices (1 x 36) np.array
            normals (3 x 24) np.array
        """

        # setup vertices
        vertices = np.empty((3,24), dtype=np.float32)
        # map vertex number to corner number (CCW around faces)
        vert_lut = [0, 1, 2, 3, # front
                    1, 5, 6, 2, # right
                    5, 4, 7, 6, # back
                    4, 0, 3, 7, # left
                    3, 2, 6, 7, # top
                    4, 5, 1, 0] # bottom
        corners = self.corners()
        for i in range(24):
            vertices[:,i] = corners[:,vert_lut[i]]

        # setup faces
        j=0 # iterates over vert_lut
        indices = np.empty((36,), dtype=np.uint32)

        for i in range(6): # loop over faces - fr, rt, bk, lf, top, bot
            if normals_out: # CCW order
                indices[i*6:i*6+6] = [j, j+1, j+2, j, j+2, j+3]
            else: # inward -> CW order
                indices[i*6:i*6+6] = [j, j+2, j+1, j, j+3, j+2]
            j+=4

        # setup normals
        normals = np.empty((3,24),dtype=np.float32)
        dir = [(0,0,1), (1,0,0), (0,0,-1), (-1,0,0), (0,1,0), (0,-1,0)]
        for i in range(6):
            normals[:,i*4:i*4+4] = np.column_stack([Vector3f(*dir[i])] * 4)
        if not normals_out:
            normals = np.negative(normals)

        return (vertices, indices, normals)
