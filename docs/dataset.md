---
id: dataset
title: Dataset
sidebar_label: Dataset
---

## Description

The SUMO challenge currently operates using synthetic data.  The data
is derived from the [SUN-CG dataset](http://suncg.cs.princeton.edu/).
It is processed using the [House3D
library](https://research.fb.com/downloads/house3d-a-rich-and-realistic-3d-environment/)
to render 360-degree RGB-D images represented as cube-maps. 

## Input Format

The SUMO input format is an RGB-D image represented as a cube-map.
The cube-map is stored in a multi-page TIFF file, with the color image
and the depth image stored separately.  Each image is 6K x 1K pixels
in size.  The color image is 8 bits per color channel.  The depth
image is actually stored as inverse depth.  The cube-map faces are
stored in the following order: back, left, front, right, top, bottom.

## Output Format

The SUMO output format is a directory containing an xml file and, for
the voxel and mesh performance tracks, a set of additional files
describing the element geometry.  The format of the xml is specified
by an xsd file.  Here is an example xml scene file.  For the voxel
track, the elements are represented by voxel grids in hdf5 format,
each in a separate file.  For the mesh track, the elements are
represented by textured meshes in glb format.

## Categories

The categories for the SUMO challenge elements are the same as those
in the SUN-CG fine-grained class list.  


## Software Download

The SUMO Challenge software includes Python code to read the SUMO input format, write the output format, and compute the evaluation metrics for a given scene.  The software is still under development.  It will be made available as open source on Github when the challenge is officially launched in July.


## Dataset Download

The SUMO dataset is still under development.  It will be made
available for download when the challenge is officially launched in July.
