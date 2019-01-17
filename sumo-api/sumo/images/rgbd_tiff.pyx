"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Read and write color and depth pairs from/to a single tiff file.
"""

cimport numpy as np
import numpy as np
from cython.operator cimport dereference, postincrement
from libc.stdint cimport int64_t
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.utility cimport pair

import sumo.geometry.inverse_depth as id
from sumo.opencv.wrap cimport CV_8UC1
from sumo.opencv.wrap cimport (
    array_from_mat3b, array_from_mat1b, array_from_mat1w)
from sumo.opencv.wrap cimport Mat1b, Mat1w, Mat3b, Mat
from sumo.opencv.wrap cimport mat3b_from_array, mat1w_from_array

cdef extern from "sumo/images/RgbdTiff.h" namespace "sumo":
  # Enum only to be used by this wrapper but no other python / cython
  cpdef enum TiffPageType:
    RGB,
    InverseDepth,
    Category,
    Instance

  ctypedef struct TiffMetadata:
    int64_t version
    float depthNearPlane

  ctypedef map[TiffPageType, Mat*] TiffPageMap
  cdef void readRgbdTiff(const string&, TiffPageMap* const,
        TiffMetadata* const) except +
  cdef void writeRgbdTiff(const TiffPageMap&, const string&,
        const TiffMetadata&) except +

ctypedef pair[TiffPageType, Mat*] TiffPageMapEntry

# This is the "public" facing enum to be used by python / cython code
class MultiImagePageType:
  RGB = 'rgb'
  Depth = 'depth'
  InverseDepth = 'invdepth'
  Category = 'category'
  Instance = 'instance'

# This map doesn't include depth or inverse depth due to special handling
kTiffToMultiImagePageType = {
  TiffPageType.RGB: MultiImagePageType.RGB,
  TiffPageType.Category: MultiImagePageType.Category,
  TiffPageType.Instance: MultiImagePageType.Instance
}

kMultiImageToTiffPageType = {
  val: key
  for key, val in kTiffToMultiImagePageType.items()
}


def load(unicode path):
  """Read from a single multi-page tiff with color and inverse depth.  Note that
     while images are always serialized as inverse depth, it is always converted
     to depth when loading
     Inputs:
         path (unicode) - tiff file path to read
     Returns map of MultiImagePageType to np.ndarray"""
  cdef TiffPageMap page_map_cv
  cdef TiffMetadata meta
  for page_type in TiffPageType:
    page_map_cv.insert(TiffPageMapEntry(page_type, new Mat()))
  py_byte_string = path.encode('UTF-8')
  readRgbdTiff(py_byte_string, &page_map_cv, &meta)

  page_map = {}
  for it in page_map_cv:
    if it.second[0].empty():
      continue
    if it.first == TiffPageType.RGB:
      if it.second[0].channels() != 3:
        raise ValueError("rgbd_tiff.load RGB does not have 3 channels")
      else:
        page_map[MultiImagePageType.RGB] = array_from_mat3b(<Mat3b>it.second[0])
    elif it.first == TiffPageType.InverseDepth:
      if it.second[0].channels() != 1:
        raise ValueError("rgbd_tiff.load Inverse depth does not have 1 channels")
      if it.second[0].type() == CV_8UC1:
        inv_depth_uint8 = array_from_mat1b(<Mat1b>it.second[0])
        # Since this is the 8-bit representation, assume we need to convert
        # from the 1m near plane to our current default near plane value
        page_map[MultiImagePageType.Depth] = id.uint16_of_uint8_inverse_depth_map(inv_depth_uint8, near=1.0)
        meta.depthNearPlane = <float>id.DEFAULT_NEAR
      else:
        page_map[MultiImagePageType.Depth] = array_from_mat1w(<Mat1w>it.second[0])
    elif it.first == TiffPageType.Category or it.first == TiffPageType.Instance:
      page_map[kTiffToMultiImagePageType[it.first]] = array_from_mat1w(<Mat1w>it.second[0])
    del it.second

  inv_depth = page_map[MultiImagePageType.Depth]
  assert inv_depth.ndim == 2
  # Convert inverse depth to depth using near plane from file metadata
  depth = id.depth_image_of_inverse_depth_map(inv_depth, meta.depthNearPlane)
  page_map[MultiImagePageType.Depth] = depth
  del inv_depth

  return page_map

def _validate(page_map_np):
  rgb_np = page_map_np[MultiImagePageType.RGB]
  if rgb_np.ndim!=3 or rgb_np.shape[2]!=3 or rgb_np.dtype!=np.uint8:
      raise ValueError("rgbd_tiff.save expects uint8 h*w*3 rgb image")

  if MultiImagePageType.Depth in page_map_np:
      depth_np = page_map_np[MultiImagePageType.Depth]
      if depth_np.ndim!=2 or depth_np.dtype!=np.float32:
          raise ValueError("rgbd_tiff.save expects float32 h*w depth image")
  elif MultiImagePageType.InverseDepth in page_map_np:
      invdepth_np = page_map_np[MultiImagePageType.InverseDepth]
      if invdepth_np.ndim != 2 or invdepth_np.dtype != np.uint16:
          raise ValueError("rgbd_tiff.save expects uint16 h*w inv depth image")

def save(page_map_np, unicode path, near=id.DEFAULT_NEAR):
  """Write to single multi-page tiff with color and inverse depth.
     Inputs:
        page_map_np - maps MultiImagePageType -> numpy array, contains all
            images to be written
        path (unicode) - output tiff file path
        near (float) - see sumo.geometry.inverse_depth"""
  _validate(page_map_np)

  cdef np.ndarray inv_depth
  cdef TiffPageMap page_map
  for page_type, img_np in page_map_np.items():
    if page_type == MultiImagePageType.Depth:
      inv_depth = id.inverse_depth_map_of_depth_image(img_np, near)
      page_map[TiffPageType.InverseDepth] = <Mat*>mat1w_from_array(inv_depth)
    elif page_type == MultiImagePageType.InverseDepth:
      # Here we use MultiImagePageType.Depth because the underlying cpp code
      # is already expecting inverse depth for this key, while the input to
      # this function does not
      page_map[TiffPageType.InverseDepth] = <Mat*>mat1w_from_array(img_np)
    elif page_type == MultiImagePageType.RGB:
      page_map[kMultiImageToTiffPageType[page_type]] = \
        <Mat*>mat3b_from_array(img_np)
    elif page_type == MultiImagePageType.Category or \
         page_type == MultiImagePageType.Instance:
      page_map[kMultiImageToTiffPageType[page_type]] = \
      <Mat*>mat1w_from_array(img_np)
    else:
      raise ValueError("rgbd_tiff.write unsupported page type")

  cdef TiffMetadata meta
  meta.depthNearPlane = <float>near;

  writeRgbdTiff(page_map, path.encode('UTF-8'), meta)

  it = page_map.begin()
  while it != page_map.end():
    del dereference(it).second
    postincrement(it)
