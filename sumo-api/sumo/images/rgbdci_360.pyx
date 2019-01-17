"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Represents a 360 rgbdci image (rgb, range, category, instance).
"""

from matplotlib import pyplot as plt
import numpy as np

from sumo.images.rgbd_tiff import MultiImagePageType
import sumo.images.rgbd_tiff as rgbd_tiff
import sumo.geometry.inverse_depth as id
from sumo.threedee.point_cloud cimport PointCloud
from sumo.threedee.point_cloud import PointCloud
from sumo.opencv.wrap cimport Mat1f, mat1f_from_array
from sumo.opencv.wrap cimport Mat3b, mat3b_from_array

ctypedef np.uint8_t Uint8
ctypedef np.float32_t Float32

class Rgbdci360(object):
    def __init__(self, rgb, range, category, instance):
        """
        Constructor

        Inputs:
          rgb (np.ndarray h*w*3) - color
          range (np.ndarray h*w) - range
          category (np.ndarray h*w) - category id
          instance (np.ndarray h*w) - instance id

        Exceptions:
        TypeError - if any of the component images have the wrong type
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise TypeError("rgb has wrong dimensions or type")
        if range.ndim != 2 or range.dtype != np.float32:
            raise TypeError("range has wrong dimensions or type")
        if category.ndim != 2 or category.dtype != np.uint16:
            raise TypeError("category has wrong dimensions or type")
        if instance.ndim != 2 or instance.dtype != np.uint16:
            raise TypeError("instance has wrong dimensions or type")

        self.rgb = rgb
        self.range = range
        self.category = category
        self.instance = instance

    def save(self, path, **options):
        """
        Save as multipage TIFF file.

        Inputs:
            path (str) - output file path
            **options - kwargs for tiff writer
        """
        page_map = {
            MultiImagePageType.RGB: self.rgb,
            MultiImagePageType.Depth: self.range,
            MultiImagePageType.Category: self.category,
            MultiImagePageType.Instance: self.instance
        }
        rgbd_tiff.save(page_map, path, **options)

    @classmethod
    def load(cls, path, **options):
        """
        Read from multi-page TIFF file.

        Inputs:
            path (string) - input file path
            **options - kwargs for tiff reader
        """
        page_map = rgbd_tiff.load(path, **options)
        rgb = page_map[MultiImagePageType.RGB]
        range = page_map[MultiImagePageType.Depth]
        category = page_map[MultiImagePageType.Category]
        instance = page_map[MultiImagePageType.Instance]
        return cls(rgb, range, category, instance)

    def show(self, figsize=(12, 6), category_lut=None):
        """
        Show using matplotlib.  Does not call plt.show().

        Inputs:
            figsize (tuple - (width, height))
              where width (int), height (int)  are figure size in pixels
            category_lut (nx1 np.array) - look up table where each row index
                contains the RGB values for the category id corresponding to
                that index.	
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(4, 1, 1)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(self.rgb)
        ax.set_title('RGB')
        ax = fig.add_subplot(4, 1, 2)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        inverse_range = id.inverse_depth_map_of_depth_image(self.range)
        plt.imshow(inverse_range, cmap='gray')
        ax.set_title('Inverse range')
        ax = fig.add_subplot(4, 1, 3)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(self.category if category_lut is None else category_lut[self.category])
        ax.set_title('Category')
        ax = fig.add_subplot(4, 1, 4)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(self.instance)
        ax.set_title('Instance')
        return fig


    def create_point_cloud(self, bool all_points=False):
        """Creates point cloud in camera frame."""
        if self.rgb.shape[0:2] != self.range.shape[0:2]:
            raise ValueError("create_point_cloud needs rgb and range sizes to agree.")

        return create_point_cloud_(self.rgb, self.range, all_points)


#---------------
# End of public interface

cdef create_point_cloud_(np.ndarray[Uint8, ndim=3] rgb,
                         np.ndarray[Float32, ndim=2] range,
                         bool all_points):
    """ Wrapper for C++ version.
        Keyword arguments:
            rgb -- h*w*3 numpy array with 3-channel uint8 RGB image
            range -- h*w numpy array with 1-channel float32 range image
            all_points -- return a cloud with all points, invalid points are
                          0,0,0
    """
    cdef Mat3b* rgb_cv = mat3b_from_array(rgb)
    cdef Mat1f* range_cv = mat1f_from_array(range)

    # Create PointCloud instance and take posession of C++ instance
    wrapper = <PointCloud>PointCloud.__new__(PointCloud)
    wrapper._c_ptr = createPointCloud(rgb_cv[0], range_cv[0], all_points)

    # De-allocate OpenCV objects, underlying memory remains with numpy
    del range_cv
    del rgb_cv

    # Return python point cloud
    return wrapper
