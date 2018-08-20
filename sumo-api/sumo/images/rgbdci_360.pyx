"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Represents a 360 rgbdci image (rgb, range, category, instance).
"""

from matplotlib import pyplot as plt
import numpy as np

from sumo.images.rgbd_tiff import MultiImagePageType
import sumo.images.rgbd_tiff as rgbd_tiff
import sumo.geometry.inverse_depth as id

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
        '''
        Read from multi-page TIFF file.

        Inputs:
            path (string) - input file path
            **options - kwargs for tiff reader
        '''
        page_map = rgbd_tiff.load(path, **options)
        rgb = page_map[MultiImagePageType.RGB]
        range = page_map[MultiImagePageType.Depth]
        category = page_map[MultiImagePageType.Category]
        instance = page_map[MultiImagePageType.Instance]
        return cls(rgb, range, category, instance)

    def show(self, figsize=(12, 6)):
        """
        Show using matplotlib.  Does not call plt.show().

        Inputs:
            figsize (tuple - (width, height))
              where width (int), height (int)  are figure size in pixels
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
        plt.imshow(self.category)
        ax.set_title('Category')
        ax = fig.add_subplot(4, 1, 4)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(self.instance)
        ax.set_title('Instance')
        return fig
