---
id: participate
title: Participate in the SUMO Challenge
sidebar_label: Participate
---

## Rules

By participating in the SUMO Challenge, you agree to be bound by the
[official SUMO Challenge Contest
Rules](https://sumochallenge.org/en/sumo-challenge-official-rules.pdf)

## Tracks

The SUMO Challenge is organized into three performance tracks based on
the output representation of the scene.  A scene is represented as a
collection of elements, each of which models one object in the scene
(e.g., a wall, the floor, or a chair).  An element is represented in one
of three increasingly descriptive representations: bounding box, voxel
grid, or surface mesh. All aspects of a scene are modeled using the
same representation.  

### Bounding Boxes Track

In the bounding boxes track, a scene is represented by a collection of
oriented bounding boxes.  This is similar to the SUN RGB-D Object
Detection Challenge.

### Voxels Track

In the voxels track, a scene is represented by a collection of oriented
voxel grids.

### Meshes Track

In the meshes track, a scene is represented by a collection of textured
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

## Prizes 

* 1st prize - winner of mesh track: $2,500 in cash + Titan X GPU
* 2nd prize - winner of voxel track: $2,000 in cash + Titan X GPU
* 3rd prize - winner of bounding box track: $1,500 in cash + Titan X GPU

See the [official SUMO Challenge Contest
Rules](https://sumochallenge.org/en/sumo-challenge-official-rules.pdf)
for details.

## How to Participate

1. Familiarize yourself with the input and output formats.
2. Download the SUMO software and the data set.  See the [data set
page](https://sumochallenge.org/docs/dataset.html) for details.
3. Develop your algorithm.
4. Submit your results using EvalAI.


## Leaderboard

The leaderboard will be hosted by EvalAI and is still under development.