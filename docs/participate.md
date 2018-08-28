---
id: participate
title: Participate in the SUMO Challenge
sidebar_label: Participate
---

## Tracks

The SUMO challenge is organized into three performance tracks based on
the output representation of the scene.  A scene is represented as a
collection of elements, each of which models one object in the scene
(e.g., a wall, the floor, or a chair).  An element is represented in one
of three increasingly descriptive representations: bounding box, voxel
grid, or surface mesh. All aspects of a scene are modeled using the
same representation.  

### Bounding Box Track

In the bounding box track, a scene is represented by a collection of
oriented bounding boxes.  This is similar to the SUN RGB-D Object
Detection Challenge.

### Voxel Track

In the voxel track, a scene is represented by a collection of oriented
voxel grids.

### Mesh Track

In the mesh track, a scene is represented by a collection of textured
surface meshes.

## Metrics

The SUMO evaluation metrics focus on the four aspects of the
representation: geometry, appearance, semantics, and perceptual
(GASP).  These metrics are based on best practices from challenges and
peer reviewed papers for related tasks. Geometry encompasses object
shape accuracy and pose error, appearance measures diffuse reflection
error, semantics captures class label precision, and perceptual
metrics measure the accuracy of the model according to human
perception.  The evaluation metrics are described in detail in the
[SUMO Challenge white paper](https://sumochallenge.org/en/sumo-white-paper.pdf)

## Prizes (Tentative)

* 1st prize - winner of mesh track: $2,500 in cash + Titan X GPU
* 2nd prize - winner of voxel track: $2,000 in cash + Titan X GPU
* 3rd prize - winner of bounding box track: $1,500 in cash + Titan X GPU

Note: Prizes and Prize availability have not been finalized and are
subject to change.  Not all challenge winners may be eligible for
prizes.  A final description of prizes and applicable terms and
conditions will be posted here prior to the
start of the SUMO challenge.

## How to Participate

1. Familiarize yourself with the input and output formats.
2. Download the SUMO software and the data set
3. Develop your algorithm.
4. Submit your results using EvalAI.

## Software Download

The SUMO Challenge software includes Python code to read the SUMO input format, write the output format, and compute the evaluation metrics for a given scene.  The software is still under development.  It will be made available as open source on Github when the challenge is officially launched in July.

## Leaderboard

Once the challenge is launched, the leaderboard will be hosted by EvalAI.