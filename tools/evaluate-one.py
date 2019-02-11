#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Evaluate one scene.
"""

import argparse
import os.path

from sumo.metrics.evaluator import Evaluator
from sumo.metrics.bb_evaluator import BBEvaluator
from sumo.metrics.mesh_evaluator import MeshEvaluator
from sumo.metrics.voxel_evaluator import VoxelEvaluator
from sumo.semantic.project_scene import ProjectScene


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one scene against ground truth"
        )
    parser.add_argument("gt_scene", help="path to ground truth scene")
    parser.add_argument("sub_scene", help="path to submitted scene")
    args = parser.parse_args()

    (gt_dir, gt_scene_name) = os.path.split(args.gt_scene)
    (sub_dir, sub_scene_name) = os.path.split(args.sub_scene)
    
    gt_scene = ProjectScene.load(gt_dir, gt_scene_name)
    sub_scene = ProjectScene.load(sub_dir, sub_scene_name)
    if gt_scene.project_type != sub_scene.project_type:
        raise ValueError('Submission and ground truth project must be the same type.')

    evaluator_lut = {
        "meshes" : MeshEvaluator,
        "voxels" : VoxelEvaluator,
        "bounding_box" : BBEvaluator,
        }
    eval_class = evaluator_lut[gt_scene.project_type]
    evaluator = eval_class(sub_scene, gt_scene, Evaluator.default_settings())
    result = evaluator.evaluate_all()

    print("Shape Score (1 is best): {}\n\
Translation Error (0 is best): {}\n\
Rotation Error (0 is best): {}\n\
Semantics Score (1 is best): {}\n\
Perceptual Score (1 is best): {}\n".format(
    result["shape_score"],
    result["translation_error"],
    result["rotation_error"],
    result["semantics_score"],
    result["perceptual_score"]))


if __name__ == "__main__":
    main()
