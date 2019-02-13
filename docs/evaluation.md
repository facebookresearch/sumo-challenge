---
id: evaluation
title: Evaluating Your Solution
sidebar_label: Evaluation
---

## Evaluation

```
from sumo.metrics.bb_evaluator import BBEvaluator
from sumo.metrics.mesh_evaluator import MeshEvaluator
from sumo.metrics.voxel_evaluator import VoxelEvaluator
from sumo.metrics.evaluator import Evaluator
from sumo.semantic.project_scene import ProjectScene

bb_ground_truth = ProjectScene.load(data_path, 'bounding_box_sample')
bb_submission = ProjectScene.load(data_path, 'bounding_box_sample')
bb_settings = Evaluator.default_settings()
bb_settings.categories = ['wall', 'floor', 'ceiling', 'sofa', 'coffee_table']
bb_evaluator = BBEvaluator(bb_submission, bb_ground_truth, bb_settings)
voxel_ground_truth = ProjectScene.load(data_path, 'voxels_sample')
voxel_submission = ProjectScene.load(data_path, 'voxels_sample')
voxel_settings = Evaluator.default_settings()
voxel_settings.categories = ['wall', 'floor', 'ceiling', 'sofa', 'coffee_table']
voxel_settings.density = 100
voxel_evaluator = VoxelEvaluator(voxel_submission, voxel_ground_truth, voxel_settings)
mesh_ground_truth = ProjectScene.load(data_path, 'meshes_sample')
mesh_submission = ProjectScene.load(data_path, 'meshes_sample')
mesh_settings = Evaluator.default_settings()
mesh_settings.categories = ['wall', 'floor', 'ceiling', 'sofa', 'coffee_table']
mesh_settings.density = 100
mesh_evaluator = MeshEvaluator(mesh_submission, mesh_ground_truth, mesh_settings)
bb_evaluator.evaluate_all()
voxel_evaluator.evaluate_all()
mesh_evaluator.evaluate_all()
```