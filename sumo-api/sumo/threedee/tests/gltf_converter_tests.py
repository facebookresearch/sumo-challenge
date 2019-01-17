#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


GLTF unit tests.
"""

import os
import shutil
import tempfile
import unittest

from libfb.py import parutil
from sumo.threedee.glb_converter import gltf2glb
from sumo.threedee.gltf import TinyGLTF


TEST_PATH = parutil.get_file_path("sumo/threedee/test_data")


class TestGlbConverter(unittest.TestCase):
    def setUp(self):
        """Create instance and get example object file path."""
        self.tiny = TinyGLTF()
        self.temp_directory = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up: remove temporary outout directory."""
        shutil.rmtree(self.temp_directory)

    def test_glb_converter(self):
        """Check that it can load the sample gltf object and write as glb."""
        input_filename = os.path.join(TEST_PATH, "Cube.gltf")
        output_filename = os.path.join(self.temp_directory, "Test.glb")
        gltf2glb(input_filename, output_filename)
        # Check the existence of glb
        self.assertTrue(os.path.isfile(output_filename))

        output_filename = os.path.join(TEST_PATH, "Cube.glb")
        gltf2glb(input_filename)
        # Check the existence of glb without specifying output_filename
        self.assertTrue(os.path.isfile(output_filename))


if __name__ == "__main__":
    unittest.main()
