"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from sumo.threedee.model_sampler import ModelSampler
from sumo.threedee.voxel_grid import VoxelGrid
from sumo.threedee.compute_bbox import ComputeBbox

class Voxelizer(object):
    """
    Algorithm to voxelize a gltf model.

    Algorithm: 
    Points are randomly sampled on the surface of the model.  The
    points are binned into the voxel grid structure.  Currently, color
    is not supported. 

    TODO: Refactor VoxelGrid class to support colors and update this
    algorithm to support colors.
    """

    def __init__(self, sampling_density=625, voxel_size=0.1):
        """
        Constructor.

        Inputs:
        sampling_density (float) - number of samples per square meter
        of surface area.
        voxel_size (float) - side length of each cube-shaped voxel (meters)
        """
        self._sampling_density = sampling_density
        self._voxel_size = voxel_size

    def run(self, model):
        """
        Inputs:
        model (GltfModel) - model to voxelize

        Return:
        voxels (VoxelGrid) - voxelized <model>
        """
        sampler = ModelSampler(self._sampling_density)
        points = sampler.run(model)
        bbox = ComputeBbox().from_point_cloud(points[:,0:3].T)
        return VoxelGrid(self._voxel_size, bbox.min_corner, points[:,0:3])
            
    
        
