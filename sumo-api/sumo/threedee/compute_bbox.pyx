"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np

from sumo.threedee.box_3d import Box3d


class ComputeBbox(object):
    """ Algorithm to compute the bounding box of a geometric object.
        Currently, meshes and point clouds are supported.
    """

    def __init__(self):
        pass

    def from_point_cloud(self, points):
        """
        Compute the bounding box of point cloud <points>.

        Input:
        points (numpy array of float - 3xN)

        Return:
        Box3d object
        """

        num_points = points.shape[1]
        if num_points == 0:
            return Box3d([0, 0, 0], [0, 0, 0])
        else:
            min_corner = np.amin(points, axis=1)
            max_corner = np.amax(points, axis=1)
            return Box3d(min_corner, max_corner)

    def from_mesh(self, mesh):
        """ Compute the bounding box of <mesh>.

            Inputs:
            mesh (Mesh or TexturedMesh)

            Return: Box3d object
        """
        return self.from_point_cloud(mesh.vertices())

    def from_meshes(self, meshes):
        """ Compute the bounding box of the union of the bounding boxes
            of all meshes in the given mesh list

            Inputs:
            meshes (list of Mesh)

            Return: Box3d object
        """
        boxes = [self.from_point_cloud(mesh.vertices()) for mesh in meshes]
        return Box3d.merge(boxes)

    def from_gltf_object(self, gltf_model):
        """ Compute the bounding box of <gltf_model>.
            In particular, the resulting bounding box is the union of the
            bounding boxes of all meshes in the gltf object.

            Inputs:
            gltf_model -- GltfModel

            Return: Box3d object
        """
        return self.from_meshes(gltf_model.primitive_meshes())
