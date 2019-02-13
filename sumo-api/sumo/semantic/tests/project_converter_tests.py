#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


ProjectConverter class unit tests.
"""

import unittest

from libfb.py import parutil
from sumo.semantic.project_converter import ProjectConverter
from sumo.semantic.project_scene import ProjectScene

TEST_PATH = parutil.get_file_path("sumo/metrics/test_data")


class TestProjectConverter(unittest.TestCase):
    def test_meshes_to_voxels(self):
        """
        Conversion from meshes to voxels.  Test number of elements
        and project_type.  Does not test contents for accuracy.
        """

        meshes_model = ProjectScene.load(TEST_PATH, "meshes_sample")
        voxel_model = ProjectConverter().run(meshes_model, "voxels")

        self.assertEqual(voxel_model.project_type, "voxels")
        self.assertEqual(len(voxel_model.elements),
                         len(meshes_model.elements))
        for element in voxel_model.elements.values():
            self.assertTrue(hasattr(element, "voxels"))

    def test_mesh_to_bbox(self):
        """
        Conversion from meshes to bbox.  Test number of elements
        and project_type.  Does not test contents for accuracy.
        """

        meshes_model = ProjectScene.load(TEST_PATH, "meshes_sample")
        bbox_model = ProjectConverter().run(meshes_model, "bounding_box")

        self.assertEqual(bbox_model.project_type, "bounding_box")
        self.assertEqual(len(bbox_model.elements),
                         len(meshes_model.elements))
        for element in bbox_model.elements.values():
            self.assertTrue(hasattr(element, "bounds"))

    def test_voxel_to_bbox(self):
        """
        Conversion from voxel to bbox.  Test number of elements
        and project_type.  Does not test contents for accuracy.
        """

        voxels_model = ProjectScene.load(TEST_PATH, "voxels_sample")
        bbox_model = ProjectConverter().run(voxels_model, "bounding_box")

        self.assertEqual(bbox_model.project_type, "bounding_box")
        self.assertEqual(len(bbox_model.elements),
                         len(voxels_model.elements))
        for element in bbox_model.elements.values():
            self.assertTrue(hasattr(element, "bounds"))

    def test_invalid_conversions(self):
        """
        Make sure invalid conversions raise an error
        """
        bbox_model = ProjectScene.load(TEST_PATH, "bounding_box_sample")
        self.assertRaises(ValueError, ProjectConverter().run,
                          project=bbox_model,
                          target_type="voxels")

        self.assertRaises(ValueError, ProjectConverter().run,
                          project=bbox_model,
                          target_type="meshes")
