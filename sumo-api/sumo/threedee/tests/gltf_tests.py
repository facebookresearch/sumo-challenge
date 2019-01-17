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
from sumo.threedee.gltf import TinyGLTF
from sumo.threedee.gltf_model import GltfModel


TEST_PATH = parutil.get_file_path("sumo/threedee/test_data")


class TestTinyGLTF(unittest.TestCase):
    def setUp(self):
        """Create instance and get example object file path."""
        self.tiny = TinyGLTF()
        self.temp_directory = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up: remove temporary outout directory."""
        shutil.rmtree(self.temp_directory)

    def check_object(self, obj, num_buffers=1, num_primitive_meshes=1, num_nodes=1):
        """Assert some properties of a given object."""
        self.assertIsNotNone(obj)
        self.assertTrue(isinstance(obj, GltfModel))
        self.assertEqual(obj.num_buffers(), num_buffers)
        self.assertEqual(obj.num_primitive_meshes(), num_primitive_meshes)
        self.assertEqual(obj.num_nodes(), num_nodes)

    def test_load(self):
        """Check that we can load the sample ASCII object."""
        filename = os.path.join(TEST_PATH, "Cube.gltf")
        obj = self.tiny.load_ascii_from_file(filename)
        self.check_object(obj)


    def test_roundtrip(self):
        """Check that we can save and then load again."""
        in_filename = os.path.join(TEST_PATH, "Cube.gltf")
        obj = self.tiny.load_ascii_from_file(in_filename)
        out_filename = os.path.join(self.temp_directory, "Test.gltf")
        self.tiny.write_gltf_scene_to_file(obj, out_filename)
        # write_gltf_scene_to_file only saves model, not textures, so copy
        texture_path = os.path.join(TEST_PATH, "Cube_BaseColor.png")
        shutil.copy(texture_path, self.temp_directory)
        texture_path = os.path.join(TEST_PATH, "Cube_MetallicRoughness.png")
        shutil.copy(texture_path, self.temp_directory)
        # now read and check
        new_obj = self.tiny.load_ascii_from_file(out_filename)
        self.check_object(new_obj)

    def test_roundtrip2(self):
        """Check that we can save and then load a custom object again."""
        input_path = os.path.join(TEST_PATH, "table_3", "table_3.gltf")
        obj = self.tiny.load_ascii_from_file(input_path)
        self.check_object(obj, num_nodes=10)
        # save model
        output_path = os.path.join(self.temp_directory, "table_3_output.gltf")
        self.tiny.write_gltf_scene_to_file(obj, output_path)
        # write_gltf_scene_to_file only saves model, not textures, so copy
        texture_path = os.path.join(TEST_PATH, "table_3", "1.png")
        shutil.copy(texture_path, self.temp_directory)
        # now read and check
        new_obj = self.tiny.load_ascii_from_file(output_path)
        self.check_object(new_obj, num_nodes=10)

    def test_roundtrip3(self):
        """Check that we can save and then load an object with many textures."""
        input_path = os.path.join(TEST_PATH, "office_chair_1", "office_chair_1.gltf")
        obj = self.tiny.load_ascii_from_file(input_path)
        self.check_object(obj, 1, 15, 49)
        # save model
        output_path = os.path.join(self.temp_directory, "office_chair_1_output.gltf")
        self.tiny.write_gltf_scene_to_file(obj, output_path)
        # write_gltf_scene_to_file only saves model, not textures, so copy
        texture_path = os.path.join(TEST_PATH, "office_chair_1", "1.png")
        shutil.copy(texture_path, self.temp_directory)
        texture_path = os.path.join(TEST_PATH, "office_chair_1", "2.png")
        shutil.copy(texture_path, self.temp_directory)
        # now read and check
        new_obj = self.tiny.load_ascii_from_file(output_path)
        self.check_object(new_obj, 1, 15, 49)

    def test_minimal(self):
        """
        Check that we can create and save a minimal object.
        Loading, however, does not work as no scene is defined by default constructor.
        """
        obj = GltfModel()
        filename = os.path.join(self.temp_directory, "Minimal.gltf")
        self.tiny.write_gltf_scene_to_file(obj, filename)
        with self.assertRaises(IOError):
            self.tiny.load_ascii_from_file(filename)


if __name__ == "__main__":
    unittest.main()
