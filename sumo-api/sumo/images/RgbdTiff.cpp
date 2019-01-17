/*
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include "RgbdTiff.h"
#include "picojson.h"
#include <stdexcept>

// These constants are used for our metadata reading/writing
#define SUMO_TIFF_VERSION 1.0
#define META_TAG_VERSION "version"
#define META_TAG_NEARPLANE "near"

using namespace std;

namespace sumo {

const std::map<std::string, TiffPageType> kStrToPageType = {
  {"rgb", TiffPageType::RGB},
  // We include the "depth" -> InversDepth mapping here due to backwards
  // compatibility (we had inverse depth incorrectly named before)
  {"depth", TiffPageType::InverseDepth},
  {"inv_depth", TiffPageType::InverseDepth},
  {"category", TiffPageType::Category},
  {"semantic", TiffPageType::Category},
  {"instance", TiffPageType::Instance}
};

// Indexing matches the enum values for TiffPageType
const std::vector<std::string> kPageTypeToStr = {
  "rgb",
  "inv_depth",
  "category",
  "instance"
};

const std::string& pageTypeToStr(const TiffPageType pageType) {
  return kPageTypeToStr.at(pageType);
}

TiffPageType strToPageType(const char* pageTypeStr) {
  auto key = std::string(pageTypeStr);
  return kStrToPageType.at(key);
}

TiffPageType readTiffPageType(TIFF* tiff, uint16 channels) {
  TiffPageType pageType;
  char *data;
  if (TIFFGetField(tiff, TIFFTAG_PAGENAME, &data) != 1) {
    // Tag not found. This only happens for previous tiff formats with only
    // RGB & Depth images.  In this case, we default to setting the page type
    // by the number of channels for backwards compatibility
    pageType = channels == 3 ? TiffPageType::RGB : TiffPageType::InverseDepth;
  }
  else {
    pageType = strToPageType(data);
  }
  return pageType;
}

// Tries to read the TIFFTAG_IMAGEDESCRIPTION for JSON-formatted
// metadata.
// Arguments:
//  tiff: Opened TIFF file.  Should be reading from the first TIFF directory
//  meta: pointer to struct holding the parsed metadata from tiff
// Throws a runtime_error if the tiff trying to be read has improperly
// formatted metadata
void readMetadata(TIFF* tiff, TiffMetadata* meta) {
  assert(meta);

  char* rawValue = nullptr;
  if (!TIFFGetField(tiff, TIFFTAG_IMAGEDESCRIPTION, &rawValue)) {
    meta->version = 0;
    // This is the legacy near plane for images which don't have this tag
    meta->depthNearPlane = 1.0;
    return;
  }
  assert(rawValue != nullptr);

  // TODO: review for potential buffer overrun issues
  size_t valLen = strlen(rawValue);
  picojson::value jsonValue;
  picojson::parse(jsonValue, rawValue, rawValue + valLen);

  if (!jsonValue.is<picojson::object>()) {
    throw runtime_error("readMetadata: corrupt JSON metadata in TIFF");
  }
  const auto& jsonObj = jsonValue.get<picojson::object>();

  if (jsonObj.find(META_TAG_VERSION) == jsonObj.end()) {
    meta->version = 0;
  }
  else {
    meta->version = jsonObj.at(META_TAG_VERSION).get<int64_t>();
  }

  if (jsonObj.find(META_TAG_NEARPLANE) == jsonObj.end()) {
    // Default near plane when this tag is missing
    meta->depthNearPlane = 1.0;
  }
  else {
    auto near = jsonObj.at(META_TAG_NEARPLANE).get<double>();
    // TODO: refactor the depth manipulation code (inverse_depth.pyx) to use
    // double instead of float32, and then we can change this to double and
    // avoid this cast
    meta->depthNearPlane = static_cast<float>(near);
  }
}

void writeMetadata(TIFF*tiff, const TiffMetadata& meta) {
  // Convert metadata to JSON
  auto versionNode = picojson::value(static_cast<int64_t>(SUMO_TIFF_VERSION));
  auto nearPlaneNode =
      picojson::value(static_cast<double>(meta.depthNearPlane));
  const picojson::value::object jsonObj = {{META_TAG_VERSION, versionNode},
                                           {META_TAG_NEARPLANE, nearPlaneNode}};
  const auto jsonValue = picojson::value(jsonObj);
  std::string serializedJson = jsonValue.serialize();
  TIFFSetField(tiff, TIFFTAG_IMAGEDESCRIPTION, serializedJson.c_str());
}

// Helper function that extracts one image from a TIFF object
// and stores result in <*pageMap> (pointer to map containing all read images)
// Read uint8 3-channel or 1-channel image.  Note that the read image could
// have one of three representations:
// 1. 8-bit 3-channel RGB image
// 2. 8-bit 1-channel depth image (for backwards compatibility)
// 3. 16-bit 1-channel depth image
bool readImage(TIFF* tiff, TiffPageMap* const pageMap) {
  int status;
  uint32 width, height;
  uint16 samplesPerPixel;
  uint16 bitsPerSample;
  TIFFGetField(tiff, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
  if (samplesPerPixel != 3 && samplesPerPixel != 1) {
    throw runtime_error("readImage: unexpected channel count");
  }

  TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
  if (bitsPerSample != 8 && bitsPerSample != 16) {
    throw runtime_error("readImage: unexpected bits per sample");
  }

  TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);
  TiffPageType pageType = readTiffPageType(tiff, samplesPerPixel);
  auto pageEntry = pageMap->find(pageType);
  if (pageEntry == pageMap->end()) {
    // Skip entry here since the calling context does not care about this image
    return true;
  }

  cv::Mat* img = pageEntry->second;
  if (samplesPerPixel == 3) {
    img->create(height, width, CV_8UC3);
  }
  else if (bitsPerSample == 16) {
    // Here we know samplesPerPixel == 1
    img->create(height, width, CV_16UC1);
  }
  else {
    // samplesPerPixel == 1 && bitsPerSample == 8
    img->create(height, width, CV_8UC1);
  }

  // TODO: is stride always channels*width for new Mat image?
  size_t stride = samplesPerPixel * width * (bitsPerSample / 8);
  for (unsigned int y = 0; y < height; ++y) {
    status = TIFFReadScanline(tiff, (void*)(img->data + y * stride), y, 0);
    if (status == -1) {
      throw runtime_error("readImage: error reading scanline");
    }
  }

  return true;
}

// Reads a multi-image TIFF and populates the given map
// path:  the full path to the TIFF file to read
// pageMap: map containing empty images for each of the page types
//    that the reader cares about
void readRgbdTiff(
    const string& path,
    TiffPageMap* const pageMap,
    TiffMetadata* const meta) {
  TIFF* tiff = TIFFOpen(path.c_str(), "r");
  if (!tiff) {
    throw runtime_error("readRgbdTiff: can't open TIFF file " + path);
  }
  // Metadata is always written to first page
  readMetadata(tiff, meta);
  // Keep reading images until there are none left
  while(readImage(tiff, pageMap) && TIFFReadDirectory(tiff));
  TIFFClose(tiff);
}

// Writes a single image to the given TIFF object
void writeImage(
    TIFF* tiff,
    const cv::Mat& img,
    TiffPageType pageType,
    uint16 pageIndex,
    uint16 totalPages) {
  uint16_t bitsPerSample = img.type() == CV_8UC3 ? 8 : 16;
  TIFFSetField(tiff, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
  TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, img.cols);
  TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, img.rows);
  TIFFSetField(
      tiff, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tiff, (unsigned int)-1));
  TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
  TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, bitsPerSample);
  TIFFSetField(tiff, TIFFTAG_PAGENUMBER, pageIndex, totalPages);
  TIFFSetField(
      tiff, TIFFTAG_SAMPLESPERPIXEL, static_cast<uint16>(img.channels()));
  auto& pageName = pageTypeToStr(pageType);
  TIFFSetField(tiff, TIFFTAG_PAGENAME, pageName.c_str());
  TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);

  // TODO: is stride always correct for every Mat image?
  size_t stride = img.channels() * img.cols * (bitsPerSample / 8);
  for (int y = 0; y < img.rows; ++y) {
    TIFFWriteScanline(tiff, (void*)(img.data + y * stride), y, 0);
  }

  TIFFWriteDirectory(tiff);
}

// Write a multi-image TIFF to the path specified
// Exceptions are thrown in the following scenarios:
// - pageMap does not have entries for RGB or depth images
// - cannot open specified path for writing
void writeRgbdTiff(
    const TiffPageMap& pageMap,
    const string& path,
    const TiffMetadata& meta) {
  if (pageMap.find(TiffPageType::RGB) == pageMap.end()) {
    throw runtime_error("writeRgbdTiff: RGB image not found");
  }
  if (pageMap.find(TiffPageType::InverseDepth) == pageMap.end()) {
    throw runtime_error("writeRgbdTiff: Depth image not found");
  }
  TIFF* tiff = TIFFOpen(path.c_str(), "w");
  if (!tiff) {
    throw runtime_error("writeRgbdTiff: can't open TIFF file " + path);
  }

  // For encoding metadata, we just write to the first page to minimize
  // storage of redundant data
  writeMetadata(tiff, meta);

  // This is slightly suboptimal, but we iterate through this way to maintain
  // backwards compatibility by making the order of images consistent
  uint16 pageIndex = 0;
  for (uint16_t i = 0; i < TiffPageType::Total; i++) {
    auto pageType = static_cast<TiffPageType>(i);
    if (pageMap.find(pageType) == pageMap.end()) {
      continue;
    }
    const cv::Mat* image = pageMap.at(pageType);
    writeImage(
        tiff, *image, pageType, pageIndex, static_cast<uint16>(pageMap.size()));
    pageIndex++;
  }
  TIFFClose(tiff);
}

} // namespace sumo
