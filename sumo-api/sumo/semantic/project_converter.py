#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


Algorithm class: Convert a ProjectScene from one type to another.
"""

from copy import deepcopy

from sumo.semantic.project_object import ProjectObject
from sumo.semantic.project_object_dict import ProjectObjectDict
from sumo.semantic.project_scene import ProjectScene
from sumo.threedee.compute_bbox import ComputeBbox
from sumo.threedee.voxelizer import Voxelizer


class ProjectConverter(object):
    """
    Convert a ProjectScene from one type to another.
    The converter only supports converting from more complex types
    to less complex types.  Specifically:
    meshes -> voxels
    voxels -> bounding_box
    meshes -> bounding_box
    """

    allowed_conversions = [("meshes", "voxels"),
                           ("meshes", "bounding_box"),
                           ("voxels", "bounding_box")]

    def __init__(self):
        pass

    def run(self, project, target_type):
        """
        Convert an in-memory project to the target type

        Inputs:
        project (ProjectScene) - input project
        target_type (string) - voxels or bounding_box

        Return:
        new_project (ProjectScene) - a project with the target project type

        Exceptions:
        ValueError - if target_type is not allowed for the given input project.

        See above for allowed conversions.
        """

        if (project.project_type, target_type) not in self.allowed_conversions:
            raise ValueError("Invalid target_type ({}) for \
                project with type {}".format(target_type, project.project_type))

        new_settings = deepcopy(project.settings)
        new_elements = ProjectObjectDict()
        for element in project.elements.values():
            new_element = self.convert_element(element, target_type)
            new_elements[new_element.id] = new_element
        new_project = ProjectScene(project_type=target_type, elements=new_elements,
            settings=new_settings)

        return new_project

    def convert_element(self, element, target_type):
        """
        Convert <element> to <target_type> track.  Makes a copy of the element.

        Inputs:
        element (ProjectObject) - element to convert
        target_type (string) - destination project type

        Return
        new_element (ProjectObject) - converted element

        See above for allowed conversions.
        """
        if (element.project_type, target_type) not in self.allowed_conversions:
            raise ValueError("Invalid target_type ({}) for element with type \
                {}".format(target_type, element.project_type))

        source_type = element.project_type
        if target_type == "bounding_box":
            if source_type == "voxels":
                bounds = element.voxels.bounds()
            elif source_type == "meshes":
                bounds = ComputeBbox().from_gltf_object(element.meshes)
            else:
                raise ValueError("Invalid target type")  # this should not be possible
            new_element = ProjectObject.gen_bounding_box_object(
                id=element.id,
                bounds=bounds,
                pose=deepcopy(element.pose),
                category=element.category,
                symmetry=element.symmetry,
                score=element.score,
                evaluated=element.evaluated
            )

        elif target_type == "voxels":
            voxelizer = Voxelizer()
            voxels = voxelizer.run(element.meshes)
            new_element = ProjectObject.gen_voxels_object(
                id=element.id,
                bounds=voxels.bounds(),
                voxels=voxels,
                pose=deepcopy(element.pose),
                category=element.category,
                symmetry=element.symmetry,
                score=element.score,
                evaluated=element.evaluated
            )

        else:
            raise ValueError("Invalid target type")  # this should not be possible

        return new_element
