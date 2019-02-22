#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import matplotlib
import numpy as np
import os
import unittest

from sumo.geometry.rot3 import Rot3
import sumo.metrics.utils as utils
from sumo.semantic.project_scene import ProjectScene


matplotlib.use("TkAgg")

"""
    Test Evaluator utils functions
"""


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
        expected_q = np.array([
            0.9515485, 0.0381346, 0.1893079, 0.2392983])  # computed manually
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

    def test_visualize_mesh(self):
        visualize = False
        self.data_path = os.path.join(os.getcwd(),
            'sumo/metrics/test_data')
        self.ground_truth = ProjectScene.load(self.data_path, 'meshes_sample')
        project_object = next(iter(self.ground_truth.elements.values()))
        mesh = next(iter(project_object.meshes.primitive_meshes()))
        utils.visualize_mesh(mesh, visualize)

    def test_compute_auc_ap(self):
        det_matches = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
                       1, 0, 1, 0, 0, 0, 0, 1, 0])
        det_scores = np.array([0.01 * k for k in [88, 70, 80, 71, 54, 74, 18, 67,
            38, 91, 44, 35, 78, 45, 14, 62, 44, 95, 23, 45, 84, 43, 48, 95]])
        n_gt = 15
        ap, _, _ = utils.compute_auc_ap(det_matches, det_scores, n_gt)
        self.assertAlmostEqual(ap, 0.22346, 4)

    def test_compute_pr(self):
        det_matches = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
                       1, 0, 1, 0, 0, 0, 0, 1, 0])
        det_scores = np.array([0.01 * k for k in [88, 70, 80, 71, 54, 74, 18, 67,
            38, 91, 44, 35, 78, 45, 14, 62, 44, 95, 23, 45, 84, 43, 48, 95]])
        recall_samples = np.linspace(0, 1, 11)
        n_gt = 15
        precision, _ = utils.compute_pr(
            det_matches, det_scores, n_gt, recall_samples, True)
        ap = np.mean(precision)
        self.assertAlmostEqual(ap, 0.23809, 4)

if __name__ == "__main__":
    unittest.main()
