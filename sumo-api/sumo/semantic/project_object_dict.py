#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Data class: Constrained dictionary containing ProjectObjects.
"""

import os
import xml.etree.ElementTree as ET

from libfb.py import parutil
from sumo.base.vector import Vector3
from sumo.geometry.pose3 import Pose3
from sumo.semantic.project_object import ProjectObject
from sumo.threedee.gltf_model import GltfModel


class ProjectObjectDict(dict):
    """
    A constrained form of dictionary.  Only ProjectObjects can be stored in it.
    Keys can be anything, but it is intended they be the object IDs.  Also
    supports reading and writing meshes of stored ProjectObjects and converting
    to and from xml.
    """

    def __getitem__(self, key):
        """Lookup <key> in the dictionary and return the associated value."""
        return dict.__getitem__(self, key)

    def __setitem__(self, key, value):
        """Insert <key>/<value> pair into dictionary, first verifying
            that the <value> is of type ProjectObject."""
        assert isinstance(value, ProjectObject)
        dict.__setitem__(self, key, value)

    def almost_equal(self, other):
        """Check that two dicts are almost equal."""
        if len(self) != len(other):
            return False
        for key in self:
            if key not in other:
                return False
            if not self[key].almost_equal(other[key]):
                return False
        return True

    def save(self, path=None):
        """
        Save self to disk.  Also creates an xml Element with the appropriate tags.
        If <project_type> of the contained ProjectObjects is "bounding_box", no
        files are written - only xml Element tags are generated, so <path> is
        not used.

        Inputs:
        path (string) - path to project directory (if project_type is voxels or
           mesh).

        Return:
            Element whose tag is <elements> with appropriate sub-elements
            (ProjectObjects).
        """
        base_elem = ET.Element("elements")
        for project_object in self.values():
            base_elem.append(project_object.save(path))
        return base_elem

    @classmethod
    def load(cls, project_type, base_elem, path=None):
        """
        Factory method to create a ProjectObjectDict by reading it from disk and
        xml.
        Note: If project_type of contained ProjectObjects is bounding_box,
        no file is read and <path> is not used.
        Note: Assumes tag of base element has been verified by caller.

        Inputs:
        project_type (string) - Specifies the project type to construct (valid
           values are "bounding_box", "voxels", or "meshes")
        base_elem - An Element with appropriate sub-elements (objects).
        path (string) - Path to project directory.

        Return:
        New ProjectObjectDict

        Exceptions:
        ValueError - If sub-elements are not ProjectObjects.
       """

        # loop over sub-elements, converting and adding to new dict
        project_object_dict = ProjectObjectDict()
        for elem in base_elem:
            if elem.tag == "element":
                po = ProjectObject.load(project_type, elem, path)
                project_object_dict[po.id] = po
        return project_object_dict

    @classmethod
    def example(cls):
        """Create a simple ProjectObjectDict."""

        pose1 = Pose3()
        pose2 = Pose3(t=Vector3(2, 2, 0))
        pose3 = Pose3(t=Vector3(4, 2, 0))

        # Be explicit about unicode literal in case this code is called from python 2
        data_path = parutil.get_dir_path(
            u"sumo/threedee/test_data")

        model1 = GltfModel.load_from_glb(os.path.join(data_path, "bed.glb"))
        model2 = GltfModel.load_from_glb(os.path.join(data_path, "bed.glb"))
        model3 = GltfModel.load_from_glb(os.path.join(data_path, "bed.glb"))

        obj1 = ProjectObject.gen_meshes_object(
            meshes=model1, id="1", pose=pose1, category="bed"
        )
        obj2 = ProjectObject.gen_meshes_object(
            meshes=model2, id="2", pose=pose2, category="chair"
        )
        obj3 = ProjectObject.gen_meshes_object(
            meshes=model3, id="3", pose=pose3, category="bed"
        )

        project_object_dict = ProjectObjectDict()
        project_object_dict[obj1.id] = obj1
        project_object_dict[obj2.id] = obj2
        project_object_dict[obj3.id] = obj3

        return project_object_dict
