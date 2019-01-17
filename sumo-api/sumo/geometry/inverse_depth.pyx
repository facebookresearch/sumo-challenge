"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import sys
cimport numpy as np

# file-specific directives:
#cython: boundscheck=False
#cython: wraparound=False

# Default near plane for inverse depth representation (in meters)
DEFAULT_NEAR = 0.3
PIXEL_MAX = np.iinfo(np.uint16).max

ctypedef np.uint8_t Uint8
ctypedef np.uint16_t Uint16
ctypedef np.float32_t Float32

# We use typed memory-views as explained in
# http://docs.cython.org/en/latest/src/userguide/memoryviews.html
# In practice it is just a nice shortcut for numpy array types
ctypedef Uint8[:,:,:] Rgb8
ctypedef Float32[:,:] Depth32
ctypedef Uint8[:,:] InvDepth8
ctypedef Uint16[:,:] InvDepth16

from numpy.math cimport INFINITY

# Converting single inverse depth value to float depth, 8 and 16 bit versions

cpdef Float32 depth_of_uint8_inverse_depth(Uint8 inverse_depth, near=DEFAULT_NEAR):
    """Convert 8-bit inverse depth to float32 depth
       Kept for backwards compatibility
    """
    if inverse_depth == 0:
        return INFINITY
    else:
        return near * 255. / <Float32>inverse_depth

cpdef Float32 depth_of_inverse_depth(Uint16 inverse_depth,
        Float32 near=DEFAULT_NEAR):
    """Calculate 32-bit float depth from int 16-bit int inverse depth.
       0 becomes infinity.
    """
    if inverse_depth == 0:
        return INFINITY
    else:
        return near * PIXEL_MAX / <Float32>inverse_depth

# Converting single float depth value to inverse depth, 8 and 16 bit versions

cpdef Uint8 uint8_inverse_depth_of_depth(Float32 depth, Float32 near=DEFAULT_NEAR):
    """Calculate 8-bit int inverse depth from 32-bit float depth.
       Does not check for 0!
    """
    cdef Uint8 result;
    if depth == 0:  # 0 depth means unknown.  map to infinity.
        return 0
    elif depth <= near:
        return 255
    else:
        return <Uint8>(0.5 + near * 255 / depth)

cpdef Uint16 inverse_depth_of_depth(Float32 depth, Float32 near=DEFAULT_NEAR):
    """Calculate 16-bit int inverse depth from 32-bit float depth.
       Does not check for 0!
    """
    cdef Uint16 result;
    if depth == 0:  # 0 depth means unknown.  map to infinity.
        return 0
    elif depth <= near:
        return PIXEL_MAX
    else:
        return <Uint16>(0.5 + near * PIXEL_MAX / depth)

# whole image conversions

def uint8_inverse_depth_map_of_depth_image(Depth32 depth_image,
        Float32 near=DEFAULT_NEAR):
    """ Calculate 8-bit int inverse depth map from 32-bit float depth image.
        Keyword arguments:
            depth_image -- a 2D array of 32-bit floats representing depth
            near -- a 32-bit float "near depth" range that corresponds to 255
        Depth values <=0 will cause an exception, all depth > near * 255 will yield 0.
    """
    cdef int h = depth_image.shape[0]
    cdef int w = depth_image.shape[1]
    cdef np.ndarray inverse_depth_map = np.empty((h,w), np.uint8)
    cdef int u, v
    for v in range(h):
        for u in range(w):
            inverse_depth_map[v,u] = uint8_inverse_depth_of_depth(
                depth_image[v,u], near)
    return inverse_depth_map

def inverse_depth_map_of_depth_image(Depth32 depth_image,
        Float32 near=DEFAULT_NEAR):
    """ Calculate 16-bit int inverse depth map from 32-bit float depth image.
        Keyword arguments:
            depth_image -- a 2D array of 32-bit floats representing depth
            near -- a 32-bit float "near depth" range that corresponds to PIXEL_MAX
        Depth values <=0 will cause an exception, all depth > near * PIXEL_MAX will yield 0.
    """
    cdef int h = depth_image.shape[0]
    cdef int w = depth_image.shape[1]
    cdef np.ndarray inverse_depth_map = np.empty((h,w), np.uint16)
    cdef int u, v
    for v in range(h):
        for u in range(w):
            inverse_depth_map[v,u] = inverse_depth_of_depth(
                depth_image[v,u], near)
    return inverse_depth_map

def depth_image_of_inverse_depth_map(InvDepth16 inverse_depth_map,
        Float32 near=DEFAULT_NEAR):
    """ Calculate 32-bit float depth image from int 16-bit uint inverse depth map
        Keyword arguments:
            inverse_depth_map -- a 2D array of 16-bit uints representing inverse depth
            near -- a 32-bit float "near depth" range that corresponds to PIXEL_MAX
        Inverse depth values of 0 are mapped to float('inf'), PIXEL_MAX to 'near'.
    """
    cdef int h = inverse_depth_map.shape[0]
    cdef int w = inverse_depth_map.shape[1]
    cdef np.ndarray depth_image = np.empty((h,w), np.float32)
    cdef int u, v
    for v in range(h):
        for u in range(w):
            depth_image[v,u] = depth_of_inverse_depth(
                inverse_depth_map[v,u], near)
    return depth_image

def depth_image_of_uint8_inverse_depth_map(InvDepth8 inverse_depth_map,
        Float32 near=DEFAULT_NEAR):
    """ Calculate 32-bit float depth image from int 8-bit uint inverse depth map
        Keyword arguments:
            inverse_depth_map -- a 2D array of 8-bit uints representing inverse depth
            near -- a 32-bit float "near depth" range that corresponds to 255
        Inverse depth values of 0 are mapped to float('inf'), 255 to 'near'.
    """
    cdef int h = inverse_depth_map.shape[0]
    cdef int w = inverse_depth_map.shape[1]
    cdef np.ndarray depth_image = np.empty((h,w), np.float32)
    cdef int u, v
    for v in range(h):
        for u in range(w):
            depth_image[v,u] = depth_of_uint8_inverse_depth(
                inverse_depth_map[v,u], near)
    return depth_image

# Single helper function for backwards compatibility

cpdef Uint16 inverse_depth_from_uint8(Uint8 inverse_depth, near=DEFAULT_NEAR):
    """Convert 8-bit inverse depth to to current default inverse depth
       Used for backwards compatibility
    """
    return inverse_depth_of_depth(
        depth_of_uint8_inverse_depth(inverse_depth, near))

def uint16_of_uint8_inverse_depth_map(InvDepth8 inverse_depth_map, near=DEFAULT_NEAR):
    """Converts inverse depth image from 8-bit representation to 16-bit
       inverse_depth_map: 8-bit inverse depth image
       near: the near plane to be used for the inverse depth encoding in the
          inverse_depth_map
       returns the same depth image with a 16-bit representation instead o
          the original 8-bit representation.  The new near plane will be
          the default PIXEL_MAX
    """

    cdef int h = inverse_depth_map.shape[0]
    cdef int w = inverse_depth_map.shape[1]
    cdef np.ndarray inverse_depth = np.empty((h,w), np.uint16)
    for v in range(h):
        for u in range(w):
            inverse_depth[v,u] = inverse_depth_from_uint8(
                inverse_depth_map[v,u], near)
    return inverse_depth
