import unittest
from sumo.semantic.project_scenev2 import ProjectSceneV2
import os
from sumo.metrics.VoxelEvaluator import VoxelEvaluator
import numpy as np
from easydict import EasyDict as edict
from sumo.metrics.utils import get_obj_points_and_voxels

"""
    Test VoxelEvaluator functions
"""


class TestVoxelEvaluator(unittest.TestCase):
    def setUp(self):
        """
            Common setup for test cases
        """
        self.ground_truth = ProjectSceneV2().read(os.path.join(os.getcwd(), 'sumo/metrics/data/voxel'), 'living_room')
        self.submission = ProjectSceneV2().read(os.path.join(os.getcwd(), 'sumo/metrics/data/voxel'), 'living_room')
        self.ground_truth = [v for k, v in list(self.ground_truth.project_elements.items())]
        self.submission = [v for k, v in list(self.submission.project_elements.items())]
        self.submission = [k for k in self.submission if np.sum(
            k._bounds._max_corner - k._bounds._min_corner) > 0]
        self.ground_truth = [k for k in self.ground_truth if np.sum(
            k._bounds._max_corner - k._bounds._min_corner) > 0]
        config = edict()
        config.categories = ['empty', 'bottle', 'bowl', 'chair', 'clock', 'couch',
                             'picture', 'refrigerator', 'tv', 'wall', 'floor',
                             'ceiling']
        config.voxel_th = 2
        config.voxel_size = 0.05
        config.IoU_start = 50
        config.IoU_end = 90
        config.IoU_step = 5
        config.empty_color = 0
        config.empty_cat = 0
        self.evaluator = VoxelEvaluator(config, self.ground_truth, self.submission)

    def test_association(self):
        matched, missed, extra = self.evaluator.association(self.ground_truth, self.submission, 0.5)
        expected = len(self.ground_truth)
        result = len(matched)
        self.assertEqual(result, expected,
                         "Expected %d matches, found %d.\n" %
                         (expected, result))
        expected = 0
        result = len(missed)
        self.assertEqual(result, expected,
                         "Expected %d missed, found %d.\n" % (expected, result))
        expected = 0
        result = len(extra)
        self.assertEqual(result, expected,
                         "Expected %d found %d.\n" % (expected, result))

    def test_evaluate_geometry(self):
        shape_score, pose_score, voxel_rmsd = self.evaluator.evaluate_geometry(
            self.ground_truth, self.submission)
        expected = 1
        result = shape_score
        self.assertEqual(result, expected,
                         "Expected shape score of %.3f, found %.3f.\n" %
                         (expected, result))
        expected = 0
        result = pose_score[0]
        self.assertAlmostEqual(result, expected, 9,
                         "Expected geodesic distance of %.3f, found %.3f.\n" %
                         (expected, result))
        expected = 0
        result = pose_score[1]
        self.assertEqual(result, expected,
                         "Expected translation error of %.3f, found %.3f.\n" %
                         (expected, result))
        expected = self.evaluator.config.voxel_size
        result = voxel_rmsd
        self.assertLessEqual(result, expected,
                         "Expected Voxel RMSD less than %.3f, found %.3f.\n" %
                         (expected, result))

    def test_evaluate_appearance(self):
        color_voxel_rmsd = self.evaluator.evaluate_appearance(
            self.ground_truth, self.submission)
        expected = 1e-5
        result = color_voxel_rmsd
        self.assertLessEqual(result, expected,
                         "Expected color voxel error less than %.3f, found %.3f.\n" %
                         (expected, result))

    def test_evaluate_semantic(self):
        semantic_score = self.evaluator.evaluate_semantics(
            self.ground_truth, self.submission)
        expected = 1
        result = semantic_score
        self.assertEqual(result, expected,
                         "Expected semantic score of %.3f, found %.3f.\n" %
                         (expected, result))
