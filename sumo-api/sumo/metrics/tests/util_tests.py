"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import unittest

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sumo.semantic.project_scene import ProjectScene
import os
import numpy as np
import pymesh
from sympy import intersection
from sympy.geometry import Point, Point3D, Segment
import transforms3d
import math
from pymesh.meshutils import remove_duplicated_vertices_raw
from sumo.metrics.utils import to_surface, visualize_mesh



from sumo.geometry.rot3 import Rot3
import sumo.metrics.utils as utils

"""
    Test Evaluator utils functions
"""



# def create_box(center, w, h,l, anglexyz):
#     v1 = center + l/2 + w/2 + h/2
#     v2 = center + l/2 + w/2 - h/2
#     v3 = center + l/2 - w/2 - h/2
#     v4 = center + l/2 - w/2 + h/2
#     v5 = center - l/2 - w/2 + h/2
#     v6 = center - l/2 + w/2 + h/2
#     v7 = center - l/2 + w/2 - h/2
#     v8 = center - l/2 - w/2 - h/2
#     i1, i2, i3, i4, i5, i6, i7, i8 = range(8)
#     faces =[[i1, i2, i3], [i1, i3, i4], [i4, i3, i8], [i4, i8, i5],
#             [i5, i8, i7], [i5, i7, i6], [i6, i7, i2], [i6, i2, i1],
#             [i1, i4, i5], [i1, i5, i6], [i2, i7, i8], [i2, i8, i3]]

#     vertices = [v1, v2, v3, v4, v5, v6, v7, v8]
#     for i in range(len(vertices)):
#         for j in range(3):
#             M = transforms3d.zooms.zfdir2mat(1)
#             M = np.dot(transforms3d.axangles.axangle2mat([j==0,j==1,j==2], anglexyz[j]), M)
#             vertices[i] = np.dot(vertices[i], M.T)

#     boxmesh = pymesh.form_mesh(np.array(vertices), np.array(faces))

#     return boxmesh

class TestUtils(unittest.TestCase):

    def test_quat_matrix(self):
        rx = 10  # degrees
        ry = 20
        rz = 30  
        Rx = Rot3.Rx(math.radians(rx))
        Ry = Rot3.Ry(math.radians(ry))
        Rz = Rot3.Rz(math.radians(rz))

        # matrix -> quat
        R = Rz * Ry * Rx
        q = utils.matrix_to_quat(R.R)
        expected_q = np.array([0.9515485, 0.0381346, 0.1893079, 0.2392983])  # computed manually
        np.testing.assert_array_almost_equal(q, expected_q, 4)

        # quat -> matrix
        R2 = utils.quat_to_matrix(expected_q)
        np.testing.assert_array_almost_equal(R2, R.R, 4)

        # round trip
        R2 = utils.quat_to_matrix(q)
        np.testing.assert_array_almost_equal(R.R, R2, 4)

    def test_quat_euler(self):
        q = np.array([0.9515485, 0.0381346, 0.1893079, 0.2392983])  

        expected_rx = 10  # degrees
        expected_ry = 20
        expected_rz = 30

        e = utils.quat_to_euler(q)
        self.assertAlmostEqual(expected_rz, math.degrees(e[0]), 4)
        self.assertAlmostEqual(expected_ry, math.degrees(e[1]), 4)
        self.assertAlmostEqual(expected_rx, math.degrees(e[2]), 4)

        q2 = utils.euler_to_quat(e)
        np.testing.assert_array_almost_equal(q2, q)

    def test_matrix_euler(self):
        rx = math.radians(10)
        ry = math.radians(20)
        rz = math.radians(30)

        euler = [rz, ry, rx]
        m = utils.euler_to_matrix(euler)
        euler2 = utils.matrix_to_euler(m)

        for i in range(3):
            self.assertAlmostEqual(euler[i], euler2[i]) 
    


    # def test_iou(self):
    #     center = np.array([0.5, 0.5, 0.5])
    #     w, h, l = 1.0, 1.0, 1.0
    #     w = np.array([w, 0, 0])
    #     l = np.array([0, l, 0])
    #     h = np.array([0, 0, h])
    #     box1 = create_box(center, w, h, l, [0, 0, 0])
    #     box2 = create_box(center + np.array([0.5, 0, 0]), w, h, l , [0, 0, 0])
    #     #box2 = create_box(center, w, h, l, [math.pi/4, math.pi/4, math.pi/4])
    #     inter = pymesh.boolean(box1, box2, operation='intersection', engine='cgal')
    #     visualize = False
    #     ivert, ifaces, _ = remove_duplicated_vertices_raw(inter.vertices, inter.faces)
    #     inter = pymesh.form_mesh(ivert, ifaces)
    #     if visualize:
    #         to_surface(box1).plot('r')
    #         to_surface(box2).plot('g')
    #         to_surface(inter).plot('b')
    #         plt.show()
    #     inter = inter.volume

    # def test_visualize_mesh(self):
    #     visualize = False
    #     for obj in self.ground_truth:
    #         visualize_mesh(obj, visualize)

