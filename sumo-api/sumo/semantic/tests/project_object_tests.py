#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

ProjectObject unit tests.
"""

import shutil
import tempfile
import unittest
import xml.etree.cElementTree as ET

import numpy as np
from sumo.base.vector import Vector3f
from sumo.geometry.pose3 import Pose3
from sumo.semantic.object_symmetry import ObjectSymmetry
from sumo.semantic.project_object import ProjectObject
from sumo.threedee.box_3d import Box3d
from sumo.threedee.gltf_model import GltfModel
from sumo.threedee.voxel_grid import VoxelGrid


class TestProjectObject(unittest.TestCase):
    """Unit tests for ProjectObject class"""

    def setUp(self):
        # create sample bounding box, meshes, and voxel grid objects
        self.bounds = Box3d([-1.5, -2.2, 3], [4, 4.1, 6.5])
        textured_mesh = self.bounds.to_textured_mesh()
        self.meshes = GltfModel.from_textured_mesh(textured_mesh)
        points = np.array([[0.1, 0.1, 0.1], [1.1, 1.1, 1.1], [1.3, 1.2, 1.4]])
        self.voxels = VoxelGrid(0.5, min_corner=Vector3f(0, 0, 0), points=points)

        # and pose
        rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        trans = Vector3f(1, 1, 1)
        self.pose = Pose3(rot, trans)

        # Create temporary outout directory.
        self.temp_directory = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up: remove temporary outout directory."""
        shutil.rmtree(self.temp_directory)

    def test_constructor(self):
        """
        Test constructor.
        """
        po = ProjectObject(id="1", project_type="bounding_box", bounds=self.bounds)

        self.assertTrue(isinstance(po.pose, Pose3))
        self.assertTrue(isinstance(po.category, "".__class__))
        po.pose.assert_almost_equal(Pose3())
        self.assertTrue(po.bounds.almost_equal(self.bounds))
        self.assertIs(po.meshes, None)
        self.assertIs(po.voxels, None)
        self.assertTrue(po.category == "unknown")
        self.assertEqual(po.id, "1")
        self.assertEqual(po.symmetry, ObjectSymmetry())
        self.assertAlmostEqual(po.score, -1)
        self.assertEqual(po.evaluated, True)

    def test_almost_equal(self):
        object1 = ProjectObject.example(id="foobar")
        self.assertTrue(object1.almost_equal(object1))
        object2 = ProjectObject.example(id="not_foobar")
        self.assertFalse(object1.almost_equal(object2))

    def test_init(self):
        """Test common construction."""
        po = ProjectObject.example()

        self.assertTrue(isinstance(po.pose, Pose3))
        self.assertTrue(isinstance(po.meshes, GltfModel))
        self.assertEqual(po.meshes.num_primitive_meshes(), 1)
        self.assertEqual(po.meshes.num_nodes(), 1)
        self.assertEqual(po.meshes.num_buffers(), 1)
        self.assertEqual(po.meshes.num_images(), 2)
        self.assertTrue(isinstance(po.category, "".__class__))
        self.assertTrue(isinstance(po.symmetry, ObjectSymmetry))
        self.assertEqual(po.category, "chair")
        self.assertEqual(po.id, "1")
        self.assertAlmostEqual(po.score, 0.57)
        self.assertEqual(po.evaluated, False)

    def test_factory_methods(self):
        """Test the gen_<X> methods for the 3 project_types"""

        po = ProjectObject.gen_bounding_box_object(id="1", bounds=self.bounds)
        self.assertEqual(po.project_type, "bounding_box")
        self.assertAlmostEqual(po.bounds, self.bounds)
        self.assertEqual(po.id, "1")

        po = ProjectObject.gen_voxels_object(id="2", voxels=self.voxels)
        self.assertEqual(po.project_type, "voxels")
        self.assertAlmostEqual(po.voxels.bounds(), self.voxels.bounds())
        self.assertEqual(po.id, "2")

        po = ProjectObject.gen_meshes_object(id="3", meshes=self.meshes)
        self.assertEqual(po.project_type, "meshes")
        self.assertEqual(
            po.meshes.num_primitive_meshes(), self.meshes.num_primitive_meshes()
        )
        self.assertEqual(po.id, "3")

    def test_setting(self):
        """Test setting elements."""

        po = ProjectObject.gen_meshes_object(id="-1", meshes=self.meshes)
        po.pose = self.pose
        po.category = "table"
        po.symmetry = ObjectSymmetry.example()
        po.score = 0.23
        po.evaluated = True

        self.assertTrue(isinstance(po.pose, Pose3))
        self.assertAlmostEqual(po.pose, self.pose)
        self.assertTrue(isinstance(po.meshes, GltfModel))
        self.assertTrue(isinstance(po.category, "".__class__))
        self.assertTrue(isinstance(po.symmetry, ObjectSymmetry))
        self.assertEqual(po.category, "table")
        self.assertEqual(po.id, "-1")
        self.assertEqual(po.symmetry, ObjectSymmetry.example())
        self.assertAlmostEqual(po.score, 0.23)
        self.assertEqual(po.evaluated, True)

    def test_xml(self):
        """Test converting to and from xml."""

        s = """<element><id>floor1</id><category>floor</category><bounds>\
<corner1>0., 0., 0.</corner1><corner2>0., 0., 0.</corner2></bounds>\
<pose><translation>-131.596614 ,  -39.9279011,   92.1260558</translation>\
<rotation><c1>1., 0., 0.</c1><c2>0., 1., 0.</c2><c3>0., 0., 1.</c3></rotation>\
</pose><symmetry><x>twoFold</x><y>twoFold</y><z>fourFold</z></symmetry>\
<detectionScore>0.23</detectionScore><evaluated>True</evaluated></element>"""
        object_xml = ET.fromstring(s)
        (id, pose, category, bounds, symmetry, score, evaluated) = \
            ProjectObject._parse_xml(object_xml)
        project_object = ProjectObject.gen_bounding_box_object(
            id, bounds, pose, category, symmetry, score, evaluated
        )
        object_xml2 = project_object._to_xml()
        self.assertEqual(ET.tostring(object_xml2, encoding="unicode"), s)

    def test_meshes_io(self):
        """Test reading and writing gltf objects."""
        po = ProjectObject(
            id="foobar",
            project_type="meshes",
            meshes=self.meshes,
            pose=self.pose,
            category="chair",
        )

        object_xml = po.save(self.temp_directory)
        po2 = ProjectObject.load("meshes", object_xml, self.temp_directory)
        self.assertTrue(po2.almost_equal(po))

    def test_bounding_box_io(self):
        """
        Test reading and writing a bounding_box object.
        """

        s = """<element><id>floor1</id><category>floor</category><bounds>\
<corner1>0., 0., 0.</corner1><corner2>0., 0., 0.</corner2></bounds>\
<pose><translation>-131.596614 ,  -39.9279011,   92.1260558</translation>\
<rotation><c1>1., 0., 0.</c1><c2>0., 1., 0.</c2><c3>0., 0., 1.</c3></rotation>\
</pose><symmetry><x>twoFold</x><y>twoFold</y><z>fourFold</z></symmetry>\
<detectionScore>0.23</detectionScore><evaluated>True</evaluated></element>"""
        object_xml = ET.fromstring(s)
        project_object = ProjectObject.load("bounding_box", object_xml)
        object_xml2 = project_object.save("bounding_box")
        self.assertEqual(ET.tostring(object_xml2, encoding="unicode"), s)

    def test_voxel_io(self):
        """Test reading and writing voxel objects."""
        points = np.array([[0.1, 0.1, 0.1], [1.1, 1.1, 1.1], [1.3, 1.2, 1.4]])
        vg = VoxelGrid(0.5, min_corner=Vector3f(0, 0, 0), points=points)
        po = ProjectObject.gen_voxels_object(
            id="foobar", voxels=vg, pose=self.pose, category="chair"
        )
        self.assertIsInstance(self.temp_directory, str)
        po_xml = po.save(self.temp_directory)
        po2 = ProjectObject.load("voxels", po_xml, self.temp_directory)

        self.assertTrue(po2.almost_equal(po))

    def test_transform_pose(self):
        po = ProjectObject(id="1", bounds=self.bounds, pose=Pose3())
        po2 = po.transform_pose(self.pose)
        self.assertTrue(po2.pose.almost_equal(self.pose))


if __name__ == "__main__":
    unittest.main()
