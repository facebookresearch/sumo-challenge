#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Test MeshEvaluator functions

TODO:
More tests:
Explicitly test sample_mesh / sample_element.  Currently, this
is implicitly tested in the constructor.

Explicitly test data association.  Add test to verify that
removing an element generates a false positive and detecting an
extra element generates a missed detection.  Add test to
check that moving an element gives lowered geometric scores

Visualize sampled point cloud and colors to verify that they are
correct.
"""
import numpy as np
import os
import unittest
import time

from sumo.semantic.project_scene import ProjectScene
from sumo.metrics.evaluator import Evaluator
from sumo.metrics.mesh_evaluator import MeshEvaluator


class TestMeshEvaluator(unittest.TestCase):
    def setUp(self):
        """
        Common setup for test cases
        """
        self.data_path = os.path.join(os.getcwd(),
            'sumo/metrics/test_data')
        self.ground_truth = ProjectScene.load(self.data_path, 'meshes_sample')
        self.submission = ProjectScene.load(self.data_path, 'meshes_sample')
        self.settings = Evaluator.default_settings()
        self.settings["categories"] = [
            'wall', 'floor', 'ceiling', 'sofa', 'coffee_table']
        self.settings["density"] = 100
        self._started_at = time.time()

    def tearDown(self):
        elapsed = time.time() - self._started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 2)))

    def test_shape_similarity(self):
        """
        Verify that the shape similarity measure is producing sane outputs.
        """

        # TODO: Get deepcopy working for ProjectScene and make a simpler
        # example for faster unit test.
        evaluator = MeshEvaluator(self.submission, self.ground_truth, self.settings)

        obj1 = self.submission.elements["1069"]
        obj2 = self.ground_truth.elements["1069"]

        # verify no offset gives sim = 1
        sim = evaluator._shape_similarity(obj1, obj2)
        self.assertAlmostEqual(sim, 1, places=1)

        # verify small offset gives sim between 0 and 1
        points_orig = obj2.posed_points
        obj2.posed_points = obj2.posed_points + np.array([0.2, 0, 0, 0, 0, 0])
        sim = evaluator._shape_similarity(obj1, obj2)
        self.assertTrue(sim < 1 and sim > 0)

        # verify large offset gives sim = 0
        obj2.posed_points = obj2.posed_points + np.array([1, 0, 0, 0, 0, 0])
        sim = evaluator._shape_similarity(obj1, obj2)
        self.assertAlmostEqual(sim, 0)

        obj2.posed_points = points_orig

        shape_score = evaluator.shape_score()
        self.assertAlmostEqual(shape_score, 1, places=1)

    def test_rms_points_error(self):
        evaluator = MeshEvaluator(self.submission, self.ground_truth, self.settings)

        rms_points_error = evaluator.rms_points_error()
        self.assertTrue(rms_points_error < 0.07)

    def test_rms_points_error_empty_point_cloud(self):

        evaluator = MeshEvaluator(self.submission, self.ground_truth, self.settings)
        self.submission.elements["57"].posed_points = np.zeros((0, 0), dtype=np.int64)

        rms_points_error = evaluator.rms_points_error()
        self.assertTrue(rms_points_error < 0.07)

    def test_rms_color_error(self):
        evaluator = MeshEvaluator(self.submission, self.ground_truth, self.settings)

        rms_color_error = evaluator.rms_color_error()
        self.assertTrue(rms_color_error < 25)
        # Note: All the test objects are textured.  The variation is significant.
        # Manually verified that color variations of 25 on average are normal
        # over short (5 cm) distances.

    def test_semantics_score(self):
        evaluator = MeshEvaluator(self.submission, self.ground_truth, self.settings)
        semantics_score = evaluator.semantics_score()
        self.assertAlmostEqual(semantics_score, 1, places=1)

    def test_perceptual_score(self):
        evaluator = MeshEvaluator(self.submission, self.ground_truth, self.settings)
        perceptual_score = evaluator.perceptual_score()
        self.assertAlmostEqual(perceptual_score, 1, places=4)

    def test_evaluate_all(self):
        """
        Test the evaluate_all function by running it and checking the
        resulting metric values.  Only a subset of metrics are checked.
        """
        evaluator = MeshEvaluator(self.submission, self.ground_truth, self.settings)
        metrics = evaluator.evaluate_all()
        self.assertTrue(metrics["rms_points_error"] < 0.07)
        self.assertAlmostEqual(metrics["shape_score"], 1, places=1)

if __name__ == "__main__":
    unittest.main()
