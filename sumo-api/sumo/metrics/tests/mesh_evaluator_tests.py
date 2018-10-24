"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Test MeshEvaluator functions
"""
from easydict import EasyDict as edict
import numpy as np
import os
import unittest
import pdb

from sumo.geometry.pose3 import Pose3
from sumo.semantic.project_scene import ProjectScene
from sumo.metrics.evaluator import Evaluator
from sumo.metrics.mesh_evaluator import MeshEvaluator

class TestMeshEvaluator(unittest.TestCase):
    def setUp(self):
        """
        Common setup for test cases
        """
        self.data_path = os.path.join(os.getcwd(), 'sumo/metrics/test_data')
        self.ground_truth = ProjectScene.load(self.data_path, 'sample2')
        self.submission = ProjectScene.load(self.data_path, 'sample2')
        self.settings = Evaluator.default_settings()
        self.settings.categories = ['wall', 'floor', 'ceiling', 'sofa', 'coffee_table']

    @unittest.skip
    def test_shape_similarity(self):
        """
        Verify that the shape similarity measure is producing sane outputs.
        """
        
        # make a dummy scene
        scene = ProjectScene("meshes")

        # TODO: Get deepcopy working for ProjectScene and make a simpler example for faster unit test.
        evaluator = MeshEvaluator(self.submission, self.ground_truth, self.settings)
        
        obj1 = self.submission.elements["1069"]
        obj2 = self.ground_truth.elements["1069"]

        # verify no offset gives sim = 1
        sim = evaluator._shape_similarity(obj1, obj2)
        self.assertAlmostEqual(sim, 1)
   
        # verify small offset gives sim between 0 and 1
        points_orig = obj2.points
        obj2.points = obj2.points + np.array([0.2, 0, 0, 0, 0, 0])
        sim = evaluator._shape_similarity(obj1, obj2)
        self.assertTrue(sim < 1 and sim > 0)
 
        # verify large offset gives sim = 0
        obj2.points = obj2.points + np.array([1, 0, 0, 0, 0, 0])
        sim = evaluator._shape_similarity(obj1, obj2)
        self.assertAlmostEqual(sim, 0)

        obj2.points = points_orig
        
        shape_score = evaluator.shape_score()
        self.assertAlmostEqual(shape_score, 1)

    @unittest.skip
    def test_rms_points_error(self):
        evaluator = MeshEvaluator(self.submission, self.ground_truth, self.settings)

        rms_points_error = evaluator.rms_points_error()
        self.assertAlmostEqual(rms_points_error, 0)

    def test_rms_color_error(self):
        evaluator = MeshEvaluator(self.submission, self.ground_truth, self.settings)

        rms_color_error = evaluator.rms_color_error()
        self.assertAlmostEqual(rms_color_error, 0)
    

    #     def test_association(self):
    #     matched, missed, extra = self.evaluator.association(self.ground_truth, self.submission, 0.5)
    #     expected = len(self.ground_truth)
    #     result = len(matched)
    #     self.assertEqual(result, expected,
    #                      "Expected %d matches, found %d.\n" %
    #                      (expected, result))
    #     expected = 0
    #     result = len(missed)
    #     self.assertEqual(result, expected,
    #                      "Expected %d missed, found %d.\n" % (expected, result))
    #     expected = 0
    #     result = len(extra)
    #     self.assertEqual(result, expected,
    #                      "Expected %d found %d.\n" % (expected, result))

    # def test_evaluate_geometry(self):
    #     shape_score, pose_score, pts_rmsd = self.evaluator.evaluate_geometry(
    #         self.ground_truth, self.submission)
    #     expected = 1
    #     result = shape_score
    #     self.assertEqual(result, expected,
    #                      "Expected shape score of %.3f, found %.3f.\n" %
    #                      (expected, result))
    #     expected = 0
    #     result = pose_score[0]
    #     self.assertAlmostEqual(result, expected, 9,
    #                      "Expected geodesic distance of %.3f, found %.3f.\n" %
    #                      (expected, result))
    #     expected = 0
    #     result = pose_score[1]
    #     self.assertEqual(result, expected,
    #                      "Expected translation error of %.3f, found %.3f.\n" %
    #                      (expected, result))
    #     expected = self.evaluator.config.voxel_size
    #     result = pts_rmsd
    #     self.assertLessEqual(result, expected,
    #                      "Expected Voxel RMSD less than %.3f, found %.3f.\n" %
    #                      (expected, result))

    # def test_evaluate_appearance(self):
    #     color_pts_rmsd = self.evaluator.evaluate_appearance(
    #         self.ground_truth, self.submission)
    #     expected = 1e-5
    #     result = color_pts_rmsd
    #     self.assertLessEqual(result, expected,
    #                      "Expected color voxel error less than %.3f, found %.3f.\n" %
    #                      (expected, result))

    # def test_evaluate_semantic(self):
    #     semantic_score = self.evaluator.evaluate_semantics(
    #         self.ground_truth, self.submission)
    #     expected = 1
    #     result = semantic_score
    #     self.assertEqual(result, expected,
    #                      "Expected semantic score of %.3f, found %.3f.\n" %
    #                      (expected, result))
