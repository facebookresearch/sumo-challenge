---
id: dataset
title: Dataset
sidebar_label: Dataset
---

## Description

The SUMO challenge currently operates using synthetic data.  The data
is derived from the [SUN-CG dataset](http://suncg.cs.princeton.edu/)
The scenes are processed to produce 360-degree RGB-D images
represented as cube-maps.  The file formats are also described in
detail in the [SUMO Challenge white paper](assets/sumo-white-paper.pdf).

## Input Format

The official SUMO input format is an RGB-D image represented as a
cube-map.  Here is a [sample SUMO input file](assets/sumo-input.tif).
The cube-map is stored in a multi-page TIFF file, with the color image
and the depth image stored separately.  Each image is 6K x 1K (6144 x 1024) pixels
in size.  The cube-map faces are stored in the following order: back,
left, front, right, top, bottom.  Each face is 1K x 1K (1024 x 1024) pixels
(i.e., side length = 1024).

The color image is stored with 8 bits per color channel, in RGB order.

The range image is stored as unsigned 16 bit integers using inverse
values. A value of 0 represents infinity (i.e., unknown range), and a
maximum value (2^16-1) represents the minimum distance, which is 0.3
meters.

Range values are converted to 3D using a standard pinhole model with
a focal length of 512 (0.5 x side length) and image center at (512,
512).

In addition to the official RGBD input, we also provide optional category and
instance images 
for assisting in training.  Pixels in the category image are integer
values corresponding to element category IDs as described below.
Pixels in the instance image are integer values also.  An instance of
an object (i.e., a chair) is assigned a unique instance ID, which
matches the ID of the corresponding element in the ground truth
scene.  Category and instance images are not provided for evaluation
scenes, and your algorithm is expected to operate without the need for
these images at test time.

The multi-page TIFF contains these four images in the following order:
1. RGB
2. range
3. category
4. instance

## Output Format

The SUMO output format is a directory containing an xml file and, for
the voxel and mesh performance tracks, a set of additional files
describing the element geometry ([example scene file](assets/sumo-output.zip))The format
of the xml is specified by an [xsd file](https://sumochallenge.org/en/sumo-scene-format.xsd).  Here
is a very simple [example xml scene
file](assets/sample_output.xml).  For the voxel
track, the elements are represented by voxel grids in [hdf5
format](https://support.hdfgroup.org/HDF5/),
each in a separate file.  For the mesh track, the elements are
represented by textured meshes in [glb](https://www.khronos.org/gltf/)
format.  (Note that glb is the binary version of glTF).

## Categories

The categories for the SUMO challenge elements are a subset of those
in the SUN-CG fine-grained class list ([categories
list](assets/categories.txt)).  Three types of
categories have been removed:
1) Animate objects (e.g., human, pet).
2) Categories with fewer than 100 instances in the training data.
3) "Unknown" category.  Instances in the unknown category are primarily
box-shaped objects, which may be used to represent instances from a
variety of categories.  In the underlying annotations, these objects
are unlabeled, and in this challenge, those objects are not evaluated.



## Software Download

The SUMO Challenge software includes Python code to read the SUMO
input format, write the output format, and compute the evaluation
metrics for a given scene.  The SUMO software can be downloaded from
[GitHub](https://github.com/facebookresearch/sumo-challenge).


## Dataset Download

The dataset is large (approximately 1.8 TB) and consists of
approximately 59,000 training scenes and 360 development evaluation
scenes.  An additional set of evaluation scenes (the challenge
evaluation set) will be released shortly before the contest concludes
for determining the final rankings.

To reduce downloading problems, the input views and output scenes
have been split into five subsets each.  Downloading the data is a
simple process:
1. Since the SUMO data set is derived from the [SUN-CG data
set](http://suncg.cs.princeton.edu), it is necessary to fill out 
and submit the combined [SUN-CG/SUMO terms of use
form](https://docs.google.com/forms/d/e/1FAIpQLSc81CMMrthfgNfBS72_UxFYDVYuxcRq4jh1QiIZfKRRZxlv0Q/viewform?usp=sf_link).
3. Once we receive confirmation, we will send the link for downloading.

The data is quite large.  Please download the data only once.  
