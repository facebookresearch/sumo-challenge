---
id: overview
title: SUMO360 Overview
sidebar_label: Overview
---
The SUMO360 API is written using a combination of C++, Python, and Cython. The user-facing interface is Python only.

The code is organized into a small number of modules that support the functions of SUMO360.

#### Input:
* images - Classes for loading and accessing the input data, which is in the form of a multi-page TIFF image, stored as a cube-map.

#### Output:
* semantic - Classes for loading, saving, and manipulating SUMO scenes (i.e., ProjectScene objects)

#### Evaluation:
* metrics - Classes for evaluating SUMO solutions (in the form of ProjectScene objects) against ground truth.

#### Support:
* base - Wrapper and utility classes
* geometry - Utility classes related to 2D and 3D geometry
* threedee - Support classes for 3D computer vision


