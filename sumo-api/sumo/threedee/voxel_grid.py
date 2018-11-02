#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import h5py
import numpy as np

from sumo.base.vector import Vector3f
from sumo.threedee.box_3d import Box3d


class VoxelGrid(object):
    """
    Store regular grid of voxels.

    Attributes:
    voxel_size (float) - length of one side of a voxel (assumed square)
    min_corner (3-vector of float, read only) - (x,y,z) coordinates of the
      minimum corner of the voxel grid in local coordinates.

    The VoxelGrid is axis-aligned.  Voxels/points less than the min_corner on
    any axis are not allowed (because voxel coordinates are non-negative by
    design).

    Points can be added to the VoxelGrid, but they cannot be removed.

    Supports reading and writing to disk using HDF format (https://www.h5py.org/).

    For efficiency, only a sparse array of voxels is used.  In 3D, this is an
    Nx3 array, where each row is a set of voxel indicies (i,j,k) for an occupied
    voxel.

    TODO:
        Add ability to store color for each voxel.
        Consider using uint for storage.
        Initialize with # of voxels in each dimension.
        Check bounds to make sure linearize will not overflow integer index range.
    """

    def __init__(self, voxel_size=1, min_corner=None, points=None):
        """
        Constructor

        Inputs:
        voxel_size (float) - length of one side of a voxel (assumed square)
        min_corner (Vector3f) - the minimum corner of voxel grid in local
           coordinates (default = [0,0,0]).
        points (Nx3 vector of float) - initial 3D points to insert into the grid.

        Exceptions:
        ValueError - if a point in <points> is less than the min_corner on any
           axis is added
        ValueError - if voxel_size <= 0
        """

        if (voxel_size <= 0):
            raise ValueError("voxel_size must be > 0")
        self._scale = 1 / voxel_size  # store inverse voxel size for efficiency
        self._min_corner = min_corner if min_corner is not None else Vector3f(0, 0, 0)
        self._voxels_3d = np.zeros((0, 3))  # voxels in Nx3 array format
        self._size_x, self._size_y = 0, 0  # x, y size of voxel space

        if points is not None:
            self.add_points(points)

    @classmethod
    def from_file(cls, filename):
        """
        Create a VoxelGrid from file <filename>

        Inputs:
        filename (string) - full pathname to file.  Should have .h5 extension.

        Return:
        new VoxelGrid object

        Exceptions:
        IOError - if file cannot be read.
        """
        vg = cls()
        vg.load(filename)
        return vg

    @property
    def min_corner(self):
        """
        Return:
        min_corner (Vector3f, read only) - the minimum corner of the VoxelGrid.
        """
        return self._min_corner

    @property
    def voxel_size(self):
        """
        Return:
        voxel_size (float, read only)
        """
        return 1 / self._scale

    def add_points(self, points):
        """
        Add <points> to the VoxelGrid.

        Inputs:
        points (Nx3 vector of float) - 3D points to insert into the grid.

        Exceptions:
        ValueError - if a point less than the min_corner on any axis is added

        Algorithm:
        In order to efficiently add the points, the voxel indices (3D)
        are temporarily converted to a 1D representation and then
        converdet back to 3D indices.
        """
        # convert points to voxel coordinates
        # voxel_index = floor((point - min_corner) * scale)
        n_points = points.shape[0]
        min_corner = np.tile(self._min_corner, (n_points, 1))
        new_voxels_3d = np.floor((points - min_corner) * self._scale).astype('int32')  # Nx3
        if (np.amin(new_voxels_3d) < 0):
            raise ValueError("Point coordinates must be >= min_corner")

        # Check if voxel space dimensions have increased
        (new_size_x, new_size_y, _) = np.amax(new_voxels_3d, 0) + 1
        if (new_size_x > self._size_x) or (new_size_y > self._size_y):
            self._size_x, self._size_y = new_size_x, new_size_y

        # convert the voxel_index to 1D and merge
        old_voxels_1d = _linearize_voxels(self._voxels_3d, self._size_x, self._size_y)
        new_voxels_1d = _linearize_voxels(new_voxels_3d, self._size_x, self._size_y)
        merged_voxels_1d = np.union1d(old_voxels_1d, new_voxels_1d)
        self._voxels_3d = _vectorize_voxels(merged_voxels_1d, self._size_x, self._size_y)

    def voxel_centers(self):
        """
        Return the centers of the occupied voxels.

        Return:
        voxel_centers (Nx3 vector of float) - centers of occupied voxels.
        """
        voxel_size = self.voxel_size
        # introduce half-voxel offset to min_corner
        # note: using broadcasting
        min_corner_with_offset = self._min_corner + 0.5 * voxel_size
        return self._voxels_3d * voxel_size + min_corner_with_offset

    def bounds(self):
        """
        Compute bounding box of occupied voxels.

        Return:
        Box3d - the bounding box
        """
        centers = self.voxel_centers()
        min_corner = np.amin(centers, 0) - 0.5 * self.voxel_size
        max_corner = np.amax(centers, 0) + 0.5 * self.voxel_size

        return Box3d(corner1=min_corner, corner2=max_corner)

    def save(self, filename):
        """
        Save VoxelGrid to <filename>.

        Inputs:
        filename (string) - Full path to file.  If it does not end in .h5, this
            .h5 will be appended.

        Exceptions:
        IOError - if file cannot be written.
        """
        try:  # note: h5py does not document excption hanlding.  TODO: Verify this
            f = h5py.File(filename, "w")
        except Exception as e:
            IOError("voxel file write failed. Err: " + str(e))
        f.create_dataset("voxels_3d", shape=self._voxels_3d.shape, data=self._voxels_3d, dtype='int32')
        f.attrs["scale"] = self._scale
        f.attrs["min_corner"] = self._min_corner
        f.attrs["size_x"] = self._size_x
        f.attrs["size_y"] = self._size_y
        f.close()

    def load(self, filename):
        """
        Load VoxelGrid from <filename>.

        Inputs:
        filename (string) - Full path to file.

        Exceptions:
        IOError - if file cannot be read.
        """
        try:
            f = h5py.File(filename, "r")
        except Exception as e:
            IOError("voxel file read failed. Err: " + str(e))
        self._voxels_3d = np.array(f["voxels_3d"], np.int32)
        self._scale = f.attrs["scale"]
        self._min_corner = np.array(f.attrs["min_corner"])
        self._size_x = f.attrs["size_x"]
        self._size_y = f.attrs["size_y"]
        f.close()


#------------- End of public interface ------------------------


def _linearize_voxels(voxels_3d, size_x, size_y):
    """
    Convert 3D voxel indices into linear array of indices.

    Inputs:
    voxels_3d (np array Nx3 of int) - 3D array of voxel indices
    size_x (int) - size of voxel space in x direction (in voxels)
    size_y (int) - size of voxel space in y direction (in voxels)

    Return:
    np vector of ints - linear array of indices

    Formula is: i = z * size_y * size_x + y * size_x + x
    This does the reverse of _vectorize_voxels
    """
    linearizer = np.array([[1], [size_x], [size_x * size_y]])  # 3x1
    return np.matmul(voxels_3d, linearizer).squeeze()  # Nx3 * 3x1 => Nx1


def _vectorize_voxels(voxels_linear, size_x, size_y):
    """
    Convert linear array of voxel indices to 3D array of voxel indices.

    Inputs:
    voxels_linear (np vector of ints) - linear array of indicies
    size_x (int) - size of voxel space in x direction (in voxels)
    size_y (int) - size of voxel space in y direction (in voxels)

    Return:
    np array Nx3 of int - 3D array of voxel indices

    This does the reverse of _linearize_voxels.
    """
    (temp, x) = np.divmod(voxels_linear, size_x)
    (z, y) = np.divmod(temp, size_y)
    return np.column_stack([x, y, z])  # Nx3
