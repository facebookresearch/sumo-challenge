"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Reading and writing multi-page tiff files.
"""

cimport numpy as np
import sumo.images.rgbd_tiff as rgbd_tiff

MultiImagePageType = rgbd_tiff.MultiImagePageType


class MultiImageTiff:
    """Multi-image tiff reader and writer.

       Public attributes:
          rgb (2D np.ndarray of uint8) (read only) - RGB image, 3 channels
          range (2D np.ndarray of float32) (read only) - Range image
          category (2D np.ndarray of uint16) (read only) - Category label
          instance (2D np.ndarray of uint16) (read only) - Instance label
    """
    def __init__(self, images not None):
        """
        Constructor
            Input:
            images (dict) - Images to use. images[page]
                maps MultiImagePageType to np.ndarray
            Note: The MultiImagePageType::Depth is used for the range image.
        """
        self._page_map = images

    def save(self, path, **options):
        """
        Write a multi-page tiff

        Inputs:
        path (string) - file to save
        options - kwargs for tiff reader
        """
        rgbd_tiff.save(self._page_map, path, **options)

    @classmethod
    def load(cls, path, **options):
        """
        Read a multi-page tiff

        Inputs:
        path (string) - file to load
        options - kwargs for tiff reader

        Return:
        new MultiImageTiff instance
        """
        page_map = rgbd_tiff.load(path, **options)
        return cls(page_map)


    def _get_image(self, page_type):
        return self._page_map[page_type] if page_type in self._page_map else \
            None

    @property
    def rgb(self):
        return self._get_image(MultiImagePageType.RGB)

    @property
    def range(self):
        return self._get_image(MultiImagePageType.Depth)

    @property
    def category(self):
        return self._get_image(MultiImagePageType.Category)

    @property
    def instance(self):
        return self._get_image(MultiImagePageType.Instance)
