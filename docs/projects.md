---
id: projects
title: Working with SUMO Projects
sidebar_label: Projects
---

## Reading a ProjectScene

```
from sumo.semantic.project_scene import ProjectScene

project_scene = ProjectScene.load(path=scenes_dir, project_name=scene)
project_scene.elements
```

## Project Scene Conversion
```
from sumo.semantic.project_converter import ProjectConverter
from sumo.semantic.project_scene import ProjectScene

mesh_ground_truth = ProjectScene.load(ground_truth_dir, scene)
project_converter = ProjectConverter()
voxel_ground_truth = project_converter.run(mesh_ground_truth, 'voxels')
bb_ground_truth = project_converter.run(mesh_ground_truth, 'bounding_box')
```

## Project Object Conversion
```
from sumo.semantic.project_converter import ProjectConverter
from sumo.semantic.project_object import ProjectObject
from sumo.threedee.compute_bbox import ComputeBbox
from sumo.threedee.gltf_model import GltfModel

meshes = GltfModel.load_from_glb(glb_path)
bounds = ComputeBbox().from_gltf_object(meshes)
project_converter = ProjectConverter()
mesh_po = ProjectObject('1', project_type='meshes', bounds=bounds, meshes=meshes)
voxel_po = project_converter.convert_element(mesh_po, 'voxels')
bb_po = project_converter.convert_element(mesh_po, 'bounding_box')
```