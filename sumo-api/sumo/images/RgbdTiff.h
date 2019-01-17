/*
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#pragma once

#include <opencv2/core/core.hpp>
#include <tiffio.h>
#include <map>
#include <string>

namespace sumo {

enum TiffPageType : uint16_t {
  RGB = 0,  // 3-channel uint8
  InverseDepth = 1, // 1-channel uint16
  Category = 2, // 1-channel uint16
  Instance = 3, // 1-channel uint16
  // Used to make it easy to iterate over enum, always increment after
  // adding new types
  Total = 4
};

typedef std::map<TiffPageType, cv::Mat*> TiffPageMap;

struct TiffMetadata {
  int64_t version; ///< Version only used when reading tiff images
  float depthNearPlane; //< Depth represented in meters
};

// Load RGB and depth images from a multi-page TIFF file
// Note that this function attempts to read the TIFFTAG_IMAGEDESCRIPTION field
// for reading JSON-formatted metadata
void readRgbdTiff(
    const std::string& path,
    TiffPageMap* const pageMap,
    TiffMetadata* const meta);

// Save RGB and depth images in a multi-page TIFF file
// Note that metadata is written in JSON format to the TIFFTAG_IMAGEDESCRIPTION
// field
void writeRgbdTiff(
    const TiffPageMap& pageMap,
    const std::string& path,
    const TiffMetadata& meta);

} // namespace sumo
