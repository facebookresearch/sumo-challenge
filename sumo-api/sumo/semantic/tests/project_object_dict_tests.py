#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

ProjectObjectDict unit tests.
"""

import os
import shutil  # for rmtree
import tempfile
import unittest

from sumo.semantic.project_object import ProjectObject
from sumo.semantic.project_object_dict import ProjectObjectDict


class TestProjectObjectDict(unittest.TestCase):
    def setUp(self):
        """Create a few ProjectObjects for use in the ProjectObjectDict."""
        self.pod = ProjectObjectDict.example()
        self.temp_directory = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up: remove temporary outout directory."""
        shutil.rmtree(self.temp_directory)

    def test_insertions(self):
        """Insert some ProjectObjects into a ProjectObjectDict."""
        pod = ProjectObjectDict()

        self.assertTrue(isinstance(pod, ProjectObjectDict))

        pod["a"] = self.pod["1"]
        pod["b"] = self.pod["2"]
        pod["c"] = self.pod["3"]

        self.assertEqual(len(pod), 3)

    def test_access(self):
        """Look up a ProjectObjectDict."""
        self.assertIsInstance(self.pod["1"], ProjectObject)

    def test_delete(self):
        """Delete a ProjectObject from the ProjectObjectDict."""
        del self.pod["2"]
        self.assertEqual(len(self.pod), 2)

    def test_object_io(self):
        """Write to files and then read back in."""

        # check that writing meshes creates the directory and stores a glb file there
        orig_pod = ProjectObjectDict.example()
        path = self.temp_directory
        xml = orig_pod.save(path)
        # FIXME: Disabled check.  Only works in Python 3.
        # ET interface must be different.
        #        self.assertIsInstance(xml, ET.Element)
        self.assertTrue(os.path.isfile(os.path.join(self.temp_directory, "1.glb")))

        # check that reading meshes gives the same as what we started with
        new_pod = ProjectObjectDict.load("meshes", xml, path)
        self.assertTrue(new_pod.almost_equal(orig_pod))

        for obj in orig_pod.values():
            self.assertTrue(obj.almost_equal(new_pod[obj.id]))


if __name__ == "__main__":
    unittest.main()
