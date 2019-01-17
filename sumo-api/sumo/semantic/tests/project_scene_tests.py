#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

ProjectScene unit tests.
"""

import os
import shutil  # for rmtree
import tempfile
import unittest

from sumo.base.vector import Vector3
from sumo.semantic.project_object import ProjectObject
from sumo.semantic.project_object_dict import ProjectObjectDict
from sumo.semantic.project_scene import ProjectScene
from sumo.threedee.box_3d import Box3d


class TestProjectScene(unittest.TestCase):
    def setUp(self):
        """Create a simple test project"""
        self.elements = ProjectObjectDict.example()  # note: project_type = mesh
        # Note: We have to create a unique project pathname to pass stress testing
        # which runs jobs in parallel.
        self.temp_directory = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up: remove temporary outout directory."""
        shutil.rmtree(self.temp_directory)

    def test_bounding_box_io(self):
        """
        Save and load a bounding_box project. Also try overwriting the
        project (should fail).
        """

        project_scene = ProjectScene(project_type="bounding_box")
        bounds = Box3d([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
        po = ProjectObject(id="1", bounds=bounds, category="chair")
        project_scene.elements["1"] = po

        # test saving
        project_scene.save(path=self.temp_directory, project_name="test")
        xml_path = os.path.join(self.temp_directory, "test", "test.xml")
        self.assertTrue(os.path.isfile(xml_path))

        # test overwriting
        self.assertRaises(
            OSError, project_scene.save, path=self.temp_directory, project_name="test"
        )

        # test loading
        project_scene = ProjectScene.load(path=self.temp_directory, project_name="test")
        self.assertIsInstance(project_scene, ProjectScene)
        self.assertIsInstance(project_scene.elements, ProjectObjectDict)
        # ::: TODO: improve check with equality test on project_scene

        # Check bounding box for the first ProjectObject
        po = project_scene.elements["1"]
        self.assertTrue(po.bounds.almost_equal(bounds, atol=0.01))
        self.assertEqual(po.category, "chair")

        # Check settings
        self.assertTrue("version" in project_scene.settings)
        self.assertTrue("categories_id" in project_scene.settings)
        self.assertTrue("categories_url" in project_scene.settings)
        # Finally save again - NOTE: This is what originially exposed the bug
        # in settings
        project_scene.save(path=self.temp_directory, project_name="second_test")

    def test_nonexistent(self):
        """test invalid load"""
        self.assertRaises(
            IOError,
            ProjectScene.load,
            path=self.temp_directory,
            project_name="does-not-exist",
        )


if __name__ == "__main__":
    unittest.main()
