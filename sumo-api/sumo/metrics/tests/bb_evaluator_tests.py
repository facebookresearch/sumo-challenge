#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Test BBEvaluator functions
"""
import math
import os
import unittest
import time

from sumo.geometry.pose3 import Pose3
from sumo.geometry.rot3 import Rot3
from sumo.metrics.bb_evaluator import BBEvaluator
from sumo.metrics.evaluator import Evaluator
from sumo.semantic.project_scene import ProjectScene


class TestBBEvaluator(unittest.TestCase):
    def setUp(self):
        """
        Common setup for test cases
        """
        self.data_path = os.path.join(os.getcwd(),
            'sumo/metrics/test_data')
        self.ground_truth = ProjectScene.load(self.data_path, 'bounding_box_sample')
        self.submission = ProjectScene.load(self.data_path, 'bounding_box_sample')
        self.settings = Evaluator.default_settings()
        self.settings["categories"] = [
            'wall', 'floor', 'ceiling', 'sofa', 'coffee_table', 'beam']
        self._started_at = time.time()

    def tearDown(self):
        elapsed = time.time() - self._started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 2)))

    def test_shape_similarity(self):
        """
        Verify that the shape similarity measure is producing sane outputs.
        """

        evaluator = BBEvaluator(self.submission, self.ground_truth, self.settings)

        obj1 = self.submission.elements["51"]
        obj2 = self.ground_truth.elements["51"]

        # verify no offset gives sim = 1
        sim = evaluator._shape_similarity(obj1, obj2)
        self.assertAlmostEqual(sim, 1)

        # verify small offset gives sim between 0 and 1
        pose_orig = obj2.pose
        obj2.pose = Pose3(t=pose_orig.t + [0.1, 0, 0], R=pose_orig.R)
        sim = evaluator._shape_similarity(obj1, obj2)
        self.assertTrue(sim < 1 and sim > 0)

        # verify large offset gives sim = 0
        obj2.pose = Pose3(t=pose_orig.t + [5, 5, 5], R=pose_orig.R)
        sim = evaluator._shape_similarity(obj1, obj2)
        self.assertAlmostEqual(sim, 0)

    def test_shape_score(self):
        """
        Test class-specific shape score metric
        """

        # verify no offset gives sim = 1
        evaluator = BBEvaluator(self.submission, self.ground_truth, self.settings)
        shape_similarity = evaluator.shape_score()
        self.assertAlmostEqual(shape_similarity, 1)

        # verify that no submission gives sim = 0
        scene = ProjectScene("bounding_box")
        evaluator2 = BBEvaluator(scene, self.ground_truth, self.settings)
        semantic_score = evaluator2.shape_score()
        self.assertEqual(semantic_score, 0)

        # verify that missed detection gives sim < 1
        self.submission.elements.pop("1069")
        evaluator3 = BBEvaluator(self.submission, self.ground_truth, self.settings)
        semantic_score = evaluator3.shape_score()
        self.assertTrue(semantic_score < 1)

        # verify that extra detection gives sim < 1
        self.ground_truth.elements.pop("57")
        self.ground_truth.elements.pop("1069")
        evaluator4 = BBEvaluator(self.submission, self.ground_truth, self.settings)
        semantic_score = evaluator4.shape_score()
        self.assertTrue(semantic_score < 1)

    def test_pose_error(self):
        """
        Test rotation and translation error metric.
        """

        self.ground_truth = ProjectScene.load(self.data_path, 'bounding_box_sample2')
        self.submission = ProjectScene.load(self.data_path, 'bounding_box_sample2')
        self.settings["thresholds"] = [0.5]

        # verify that correct pose is ok
        evaluator = BBEvaluator(self.submission, self.ground_truth, self.settings)
        rotation_error, translation_error = evaluator.pose_error()
        self.assertAlmostEqual(rotation_error, 0)
        self.assertAlmostEqual(translation_error, 0)

        # verify that rotation by symmetry amount is ok
        pose_orig = self.submission.elements["1069"].pose
        new_pose = Pose3(R=pose_orig.R * Rot3.Ry(math.pi), t=pose_orig.t)
        self.submission.elements["1069"].pose = new_pose
        evaluator = BBEvaluator(self.submission, self.ground_truth, self.settings)
        rotation_error, translation_error = evaluator.pose_error()
        self.assertAlmostEqual(rotation_error, 0)
        self.assertAlmostEqual(translation_error, 0)

        # verify that rotation by non-symmetry amount give correct error
        new_pose = Pose3(R=pose_orig.R * Rot3.Ry(math.radians(10)),
                         t=pose_orig.t)
        self.submission.elements["1069"].pose = new_pose
        evaluator = BBEvaluator(self.submission, self.ground_truth, self.settings)
        rotation_error, translation_error = evaluator.pose_error()
        self.assertAlmostEqual(rotation_error, math.radians(10))
        self.assertAlmostEqual(translation_error, 0)

        # verify that translation gives translation error
        new_pose = Pose3(R=pose_orig.R,
                         t=pose_orig.t + [0.05, 0, 0])
        self.submission.elements["1069"].pose = new_pose
        evaluator = BBEvaluator(self.submission, self.ground_truth, self.settings)
        rotation_error, translation_error = evaluator.pose_error()
        self.assertAlmostEqual(rotation_error, 0)
        self.assertAlmostEqual(translation_error, 0.05)

        # verify that empty sumission gives None, None
        new_pose = Pose3(R=pose_orig.R, t=pose_orig.t + [1, 0, 0])
        self.submission.elements["1069"].pose = new_pose
        evaluator = BBEvaluator(self.submission, self.ground_truth, self.settings)
        rotation_error, translation_error = evaluator.pose_error()
        self.assertEqual(rotation_error, math.inf)
        self.assertEqual(translation_error, math.inf)

    def test_semantic_score(self):
        # We cannot test Evaluator directly.  This creates the similarity cache and
        # does data association
        evaluator = BBEvaluator(self.submission, self.ground_truth, self.settings)

        # submission and GT are exactly the same, should get mAP = 1
        semantic_score = evaluator.semantics_score()
        expected = 1
        self.assertEqual(semantic_score, expected,
                         "Expected semantic score of %.3f, found %.3f.\n" %
                         (expected, semantic_score))

        # move the coffee table by a bit to get IoU ~ 0.72.  Should
        # get 0.5 since the average precision in 5 of the 10
        # thresholds is 1 and in the other cases it is 0.
        table = self.submission.elements["1069"]
        settings = self.settings
        settings["categories"] = ["coffee_table"]
        pose_orig = table.pose
        table.pose = Pose3(t=pose_orig.t + [0.1, 0, 0], R=pose_orig.R)
        evaluator2 = BBEvaluator(self.submission, self.ground_truth, settings)
        semantic_score = evaluator2.semantics_score()
        expected = 0.5
        self.assertAlmostEqual(semantic_score, expected, 3,
                               "Expected semantic score of %.3f, found %.3f.\n" %
                               (expected, semantic_score))

        # move the coffee table by a lot to get IoU < 0.5.
        table = self.submission.elements["1069"]
        settings = self.settings
        settings["categories"] = ["coffee_table"]
        table.pose = Pose3(t=pose_orig.t + [0.5, 0, 0], R=pose_orig.R)
        evaluator2 = BBEvaluator(self.submission, self.ground_truth, settings)
        semantic_score = evaluator2.semantics_score()
        expected = 0
        self.assertAlmostEqual(semantic_score, expected, 3,
                               "Expected semantic score of %.3f, found %.3f.\n" %
                               (expected, semantic_score))

if __name__ == "__main__":
    unittest.main()
