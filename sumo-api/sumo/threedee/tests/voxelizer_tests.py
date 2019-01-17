#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


Voxelizer class unit tests.
"""

import os
import unittest

from libfb.py import parutil
from sumo.threedee.gltf_model import GltfModel
from sumo.threedee.voxelizer import Voxelizer

TEST_PATH = parutil.get_file_path("sumo/threedee/test_data")


class TestVoxelizer(unittest.TestCase):
    def test_voxelizer(self):
        """
        Test to make sure that the algorithm runs and produces some output.
        It is difficult to check the validity of the output, so we have
        some code that will allow to manually visual
        """
        input_path = os.path.join(TEST_PATH, "bed.glb")
        model = GltfModel.load_from_glb(input_path)
        voxelizer = Voxelizer()
        voxels = voxelizer.run(model)
        centers = voxels.voxel_centers()

        self.assertEqual(centers.shape[1], 3)
        self.assertTrue(centers.shape[0] > 0)

# This code will manually visualize the samples to verify that it is working.
#        ptc = PointCloud(points = centers.T)
#        ptc.write_ply(os.path.join(TEST_PATH, "bed_voxels.ply"))
