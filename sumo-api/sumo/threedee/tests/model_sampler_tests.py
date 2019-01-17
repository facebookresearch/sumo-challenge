#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


ModelSampler class unit tests.
"""

import os
import unittest

from libfb.py import parutil
from sumo.threedee.gltf_model import GltfModel
from sumo.threedee.model_sampler import ModelSampler

TEST_PATH = parutil.get_file_path("sumo/threedee/test_data")


class TestModelSampler(unittest.TestCase):
    def test_model_sampler(self):
        """
        Test to make sure that the algorithm runs and produces some output.
        It is difficult to check the validity of the output, so we have
        some code that will allow to manually visual
        """
        input_path = os.path.join(TEST_PATH, "bed.glb")
        model = GltfModel.load_from_glb(input_path)
        sampler = ModelSampler()
        samples = sampler.run(model)

        self.assertEqual(samples.shape[1], 6)
        self.assertTrue(samples.shape[0] > 0)

    def test_min_samples(self):
        """
        Test to ensure that with low sampling density, the algorithm
        still produces at least one sample per face.
        """
        input_path = os.path.join(TEST_PATH, "bed.glb")
        model = GltfModel.load_from_glb(input_path)
        sampler = ModelSampler(sampling_density=1)
        samples = sampler.run(model)

        self.assertEqual(samples.shape[1], 6)
        self.assertTrue(samples.shape[0] > 0)

        # This code will manually visualize the samples to verify that it is working.
        # ptc = PointCloud(points = samples[:,0:3].T,
        #                  colors = samples[:,3:6].T)
        # ptc.write_ply(os.path.join(TEST_PATH, "bed.ply"))
