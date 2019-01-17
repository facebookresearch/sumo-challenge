"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Represents a 360 rgbd image (rgb, range).
"""

from matplotlib import pyplot as plt
import numpy as np

from sumo.images.rgbd_tiff import MultiImagePageType
import sumo.images.rgbd_tiff as rgbd_tiff
import sumo.geometry.inverse_depth as id

class Rgbd360(object):
    def __init__(self, rgb, range):
        """
        Constructor

        Inputs:
          rgb (np.ndarray h*w*3) - color
          range (np.ndarray h*w) - range

        Exceptions:
        TypeError - if any of the component images have the wrong type
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise TypeError("rgb has wrong dimensions or type")
        if range.ndim != 2 or range.dtype != np.float32:
            raise TypeError("range has wrong dimensions or type")

        self.rgb = rgb
        self.range = range

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
        return cls(rgb, range)

    def show(self, figsize=(12, 6)):
        """
        Show using matplotlib.  Does not call plt.show().

        Inputs:
            figsize (tuple - (width, height))
              where width (int), height (int)  are figure size in pixels
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(2, 1, 1)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(self.rgb)
        ax.set_title('RGB')
        ax = fig.add_subplot(2, 1, 2)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        inverse_range = id.inverse_depth_map_of_depth_image(self.range)
        plt.imshow(inverse_range, cmap='gray')
        ax.set_title('Inverse range')
        return fig
