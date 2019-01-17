#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


GLTF unit tests.
"""

import numpy as np
import os
import shutil
import tempfile
import unittest

from libfb.py import parutil
from sumo.base.vector import Vector3f, Vector3
from sumo.threedee.gltf_model import GltfModel
from sumo.threedee.mesh import Mesh
from sumo.threedee.textured_mesh import TexturedMesh


TEST_PATH = parutil.get_file_path("sumo/threedee/test_data")


class TestGltfModel(unittest.TestCase):
    def test_deep_copy(self):
        """Check that we can copy gltf models"""

        input_path = os.path.join(TEST_PATH, "bed.glb")
        model = GltfModel.load_from_glb(input_path)
        model2 = model.deepcopy()
        self.assertEqual(model.num_primitive_meshes(), model2.num_primitive_meshes())
        model = GltfModel.example()
        self.assertNotEqual(model.num_primitive_meshes(), model2.num_primitive_meshes())

    def test_add_mesh(self):
        """Test python wrapper to go from TexturedMesh to GltfModel."""
        model = GltfModel()
        mesh = TexturedMesh.example()
        model.add_textured_primitive_mesh(mesh)
        self.assertEqual(model.num_primitive_meshes(), 1)
        self.assertEqual(model.num_nodes(), 1)
        self.assertEqual(model.num_buffers(), 1)

    def test_meshes(self):
        """Check that we can retrieve all meshes in an object."""
        input_path = os.path.join(TEST_PATH, "Cube.gltf")
        model = GltfModel.load_from_gltf(input_path)
        self.assertEqual(model.num_primitive_meshes(), 1)
        meshes = model.primitive_meshes()
        self.assertEqual(len(meshes), 1)
        self.assertIsInstance(meshes[0], TexturedMesh)
        self.assertEqual(meshes[0].num_vertices(), 36)

    def test_meshes2(self):
        """Check that we can retrieve all meshes in an object, multiple meshes."""
        input_path = os.path.join(TEST_PATH, "bed.glb")
        model = GltfModel.load_from_glb(input_path)
        self.assertEqual(model.num_primitive_meshes(), 3)
        meshes = model.primitive_meshes()
        self.assertEqual(len(meshes), 3)
        self.assertEqual(meshes[0].num_vertices(), 1189)
        self.assertLessEqual(np.max(meshes[0].indices()), meshes[0].num_vertices() - 1)
        for i in range(3):
            self.assertIsInstance(meshes[i], Mesh)
        for i in range(3):
            self.assertTrue(meshes[i].is_textured())
            self.assertIsInstance(meshes[i], TexturedMesh)

    def test_meshes_multiple_primitives(self):
        """Check we can load glb which has multiple primitives."""
        input_path = os.path.join(TEST_PATH, "bed.glb")
        model = GltfModel.load_from_glb(input_path)
        self.assertEqual(model.num_primitive_meshes(), 3)
        meshes = model.primitive_meshes()
        self.assertEqual(len(meshes), 3)
        indices = [3672, 324, 1032]
        for i in range(3):
            self.assertEqual(meshes[0].num_vertices(), 1189)
            self.assertIsInstance(meshes[i], Mesh)
            self.assertTrue(meshes[i].is_textured())
            self.assertIsInstance(meshes[i], TexturedMesh)
            self.assertEqual(meshes[i].num_indices(), indices[i])

    def test_meshes_no_base_color(self):
        """Check we can load a glb which has meshes without base color"""
        input_path = os.path.join(TEST_PATH, "blind.glb")
        model = GltfModel.load_from_glb(input_path)
        self.assertEqual(model.num_primitive_meshes(), 6)
        meshes = model.primitive_meshes()
        self.assertEqual(len(meshes), 6)
        for i in [0, 1, 3, 4]:
            self.assertIsInstance(meshes[i], TexturedMesh)
        for i in [2, 5]:
            self.assertIsInstance(meshes[i], Mesh)

    def test_example(self):
        """Test example class method."""
        model = GltfModel.example()
        self.assertEqual(model.num_primitive_meshes(), 1)
        self.assertEqual(model.num_nodes(), 1)
        self.assertEqual(model.num_buffers(), 1)

    def test_load_gltf(self):
        """Check that loading from gltf loads images."""
        input_path = os.path.join(TEST_PATH, "Cube.gltf")
        model = GltfModel.load_from_gltf(input_path)
        self.assertEqual(model.num_primitive_meshes(), 1)
        self.assertEqual(model.num_images(), 2)
        expected_size = 512 * 512
        self.assertEqual(model.image_size(0), expected_size * 4)
        self.assertEqual(model.image_size(1), expected_size * 3)
        self.assertEqual(model.image_uri(0), "Cube_BaseColor.png")
        self.assertEqual(model.image_uri(1), "Cube_MetallicRoughness.png")

    def test_load_glb(self):
        """
        Test glb loading.
        Just ensures loading returns success and that some fields are correct.
        """
        input_path = os.path.join(TEST_PATH, "bed.glb")
        model = GltfModel.load_from_glb(input_path)
        # note: these values were manually checked against the text part of
        # the glb file. The sizes are regression tests.
        self.assertEqual(model.num_primitive_meshes(), 3)
        self.assertEqual(model.num_images(), 3)
        self.assertEqual(model.image_size(0), 196608)
        self.assertEqual(model.image_size(1), 196608)
        # Note: The fact that these URIs are empty after reading from a glb
        # means that we have to make up URI's when creating a glb.
        self.assertEqual(model.image_uri(0), "0.png")
        self.assertEqual(model.image_uri(1), "1.png")

    def test_load_textured_mesh(self):
        """Check that we can load a textured mesh from a gltf file."""
        input_path = os.path.join(TEST_PATH, "Cube.gltf")
        model = GltfModel.load_from_gltf(input_path)
        mesh = model.extract_textured_primitive_mesh()
        self.assertIsInstance(mesh, TexturedMesh)
        self.assertEqual(mesh.num_vertices(), 36)
        self.assertTrue(mesh.has_dual_texture_material())


class TestGltfModelIO(unittest.TestCase):
    def setUp(self):
        """Create instance and get example object file path."""
        self.temp_directory = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up: remove temporary outout directory."""
        shutil.rmtree(self.temp_directory)

    def assert_temp_file(self, filename):
        self.assertTrue(os.path.isfile(os.path.join(self.temp_directory, filename)))

    def test_save(self):
        """Check that we can save a GltfModel instance to gltf + files."""
        model = GltfModel.example()
        model.save_as_gltf(self.temp_directory, "example")
        # Check the existence of gltf
        self.assert_temp_file("example.gltf")
        self.assert_temp_file("example.bin")
        self.assert_temp_file("0.png")
        self.assert_temp_file("1.png")

    def test_save_as_glb(self):
        """Check that we can save a GltfModel instance to glb."""
        model = GltfModel.example()
        path = os.path.join(self.temp_directory, "example.glb")
        model.save_as_glb(path)
        # Check the existence of glb
        self.assert_temp_file("example.glb")

    def test_glb_roundtrip(self):
        """Check that we can load, save, and re-load the glb."""
        # 1. Read
        input_path = os.path.join(TEST_PATH, "blind.glb")
        model = GltfModel.load_from_glb(input_path)
        self.assertEqual(model.num_primitive_meshes(), 6)
        indices = model.primitive_meshes()[0].indices()
        self.assertEqual(model.num_images(), 4)
        self.assertLessEqual(
            np.max(indices), model.primitive_meshes()[0].num_vertices() - 1
        )
        image_props = [(model.image_size(i), model.image_uri(i)) for i in range(4)]

        # 2. Save
        output_path = os.path.join(self.temp_directory, "test.glb")
        model.save_as_glb(output_path)
        self.assert_temp_file("test.glb")

        # 3. Read again
        model2 = GltfModel.load_from_glb(output_path)
        self.assertEqual(model2.num_primitive_meshes(), 6)
        indices2 = model2.primitive_meshes()[0].indices()
        np.testing.assert_array_equal(indices, indices2)
        self.assertEqual(model2.num_images(), 4)
        for i, props in enumerate(image_props):
            size, uri = props
            self.assertEqual(model2.image_size(i), size)
            self.assertEqual(model2.image_uri(i), uri)

    def test_mesh_roundtrip(self):
        """Check that we can save then read a textured mesh."""
        mesh = TexturedMesh.example()
        expected_color = mesh.base_color()[0, 0, :]
        expected_mr = mesh.metallic_roughness()[0, 0, :]

        model = GltfModel.from_textured_mesh(mesh)
        model.save_as_gltf(self.temp_directory, "dummy")

        # now load...
        input_path = os.path.join(self.temp_directory, "dummy.gltf")
        model2 = GltfModel.load_from_gltf(input_path)
        self.assertEqual(model2.num_images(), 2)
        self.assertEqual(model.image_size(0), 288)
        self.assertEqual(model.image_size(1), 72)
        self.assertEqual(model.image_uri(0), "0.png")
        self.assertEqual(model.image_uri(1), "1.png")

        # ...and check mesh.
        mesh2 = model2.extract_textured_primitive_mesh()
        self.assertEqual(mesh2.num_vertices(), 24)
        self.assertTrue(mesh.has_dual_texture_material())
        np.testing.assert_array_equal(mesh2.base_color()[0, 0, :], expected_color)
        np.testing.assert_array_equal(mesh2.metallic_roughness()[0, 0, :], expected_mr)

    def test_save_textured_mesh_as_glb(self):
        """Check that we can save a textured mesh directly to glb."""
        mesh = TexturedMesh.example()
        model = GltfModel.from_textured_mesh(mesh)
        path = os.path.join(self.temp_directory, "dummy.glb")
        model.save_as_glb(path)
        # Check the existence of glb
        self.assertTrue(os.path.isfile(path))

    def test_save_as_gltf2(self):
        """Test for material properties round trip."""
        input_path = os.path.join(TEST_PATH, "bed.glb")
        model = GltfModel.load_from_glb(input_path)
        mesh = model.extract_textured_primitive_mesh()

        model2 = GltfModel.from_textured_mesh(mesh)
        model2.save_as_gltf(self.temp_directory, "test")
        gltf_path = os.path.join(self.temp_directory, "test.gltf")
        self.assertTrue(os.path.isfile(gltf_path))

        model3 = GltfModel.load_from_gltf(gltf_path)
        mesh3 = model3.extract_textured_primitive_mesh()
        self.assertTrue(mesh3.has_same_material(mesh))

    def test_all_mesh_materials(self):
        """Round trip with model having multiple textured, non-textured meshes"""
        input_path = os.path.join(TEST_PATH, "blind.glb")
        model = GltfModel.load_from_glb(input_path)
        meshes = model.primitive_meshes()

        model2 = GltfModel()
        for mesh in meshes:
            model2.add_mesh(mesh)
        meshes2 = model2.primitive_meshes()
        self.assertEqual(len(meshes2), 6)

        model2.save_as_gltf(self.temp_directory, "test")
        gltf_path = os.path.join(self.temp_directory, "test.gltf")
        self.assertTrue(os.path.isfile(gltf_path))

        model3 = GltfModel.load_from_gltf(gltf_path)
        meshes3 = model3.primitive_meshes()
        self.assertTrue(len(meshes) == len(meshes3))

        for i in range(len(meshes)):
            self.assertTrue(meshes3[i].has_same_material(meshes[i]))

    def test_add_colored_material(self):
        mesh = TexturedMesh.example()
        model = GltfModel.from_textured_mesh(mesh)
        index = model.add_colored_material(
            "my_material", Vector3f(0.5, 0.5, 0.5), 0.0, 0.95
        )
        self.assertEqual(index, 1)
        index = model.add_colored_material(
            "another_material", Vector3f(0.75, 0.75, 0.75), 0.1, 0.7
        )
        self.assertEqual(index, 2)

    def test_update_material(self):
        mesh = TexturedMesh.example()
        model = GltfModel.from_textured_mesh(mesh)
        material = {"color" : Vector3(0.5, 0.5, 0.5), "uri": ""}
        model.update_materials("", [material])

    def test_update_textured_material(self):
        mesh = TexturedMesh.example()
        model = GltfModel.from_textured_mesh(mesh)
        self.assertEqual(model.num_images(), 2)
        uri = "Cube_BaseColor.png"
        base_dir = TEST_PATH
        material = {"color" : Vector3(0, 0, 0), "uri": uri}
        model.update_materials(base_dir, [material])
        self.assertEqual(model.num_images(), 3)


if __name__ == "__main__":
    unittest.main()
