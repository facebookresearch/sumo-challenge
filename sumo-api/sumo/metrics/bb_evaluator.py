#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Algorithm class: Evaluate a mesh track submission
"""

import numpy as np
import pymesh
from pymesh.meshutils import remove_duplicated_vertices_raw

from sumo.metrics.evaluator import Evaluator
from sumo.threedee.compute_bbox import ComputeBbox


class BBEvaluator(Evaluator):
    """
    Algorithm to evaluate a submission for the bounding box track.
    """
    def __init__(self, submission, ground_truth, settings=None):
        """
        Constructor.  Computes similarity between all elements in the
        submission and ground_truth and also computes
        data association caches.

        Inputs:
        submission (ProjectScene) - Submitted scene to be evaluated
        ground_truth (ProjectScene) - The ground truth scene
        settings (dict) - configuration for the evaluator.  See
        Evaluator.py for recognized keys and values.
        """

        # extract posed bounds and save
        # (used for IoU calcs)
        for e in submission.elements.values():
            posed_corners = e.pose.transform_all_from(e.bounds.corners())
            e.posed_bbox = ComputeBbox().from_point_cloud(posed_corners)

        for e in ground_truth.elements.values():
            posed_corners = e.pose.transform_all_from(e.bounds.corners())
            e.posed_bbox = ComputeBbox().from_point_cloud(posed_corners)

        super(BBEvaluator, self).__init__(submission, ground_truth, settings)

    def evaluate_all(self):
        """
        Computes all metrics for the submission

        Return:
        metrics (dict) - Keys/values are:
        "shape_score" : float
        "rotation_error" : float
        "translation_error" : float
        "semantics_score" : float
        "perceptual_score" : float
        """
        metrics = {}

        metrics["shape_score"] = self.shape_score()
        rotation_error, translation_error = self.pose_error()
        metrics["rotation_error"] = rotation_error
        metrics["translation_error"] = translation_error
        metrics["semantics_score"] = self.semantics_score()
        metrics["perceptual_score"] = self.perceptual_score()

        return metrics

#------------------------
# End of public interface
#------------------------

    def _shape_similarity(self, element1, element2):
        """
        Similarity function that compares the bounding boxes of
        <element1> and <element2>

        Inputs:
        element1 (ProjectObject)
        element2 (ProjectObject)

        Return:
        float - bounding box IoU (Equation 1 in SUMO white paper)
        """

        # quick intersection test.  If bounding boxes don't overlap on any single axis,
        # then the enclosed object cannot overlap
        bbox1 = element1.posed_bbox
        bbox2 = element2.posed_bbox
        for axis in range(3):
            if (bbox1.min_corner[axis] > bbox2.max_corner[axis]) or \
               (bbox2.min_corner[axis] > bbox1.max_corner[axis]):
                return 0

        box1 = _bbox2pymesh(element1)
        box2 = _bbox2pymesh(element2)
        inter = pymesh.boolean(box1, box2, operation='intersection')
        ivert, ifaces, _ = remove_duplicated_vertices_raw(inter.vertices, inter.faces)
        inter_mesh = pymesh.form_mesh(ivert, ifaces)

        # Note: pymesh may give -volume depending on surface normals
        # or maybe vertex ordering
        intersection = abs(inter_mesh.volume)
        union = abs(box1.volume) + abs(box2.volume) - intersection
        return intersection / union


def _bbox2pymesh(element):
    """
    Convert the bounding box of <element> into a pymesh Mesh in world coordinates.

    Inputs:
    element (ProjectObject)

    Return:
    pymesh.Mesh - Mesh representation of the oriented bounding box in
      world coordinates.
    """
    vertices = (element.pose * element.bounds.corners())

    i1, i2, i3, i4, i5, i6, i7, i8 = range(8)
    faces = [[i1, i2, i3], [i1, i3, i4], [i4, i3, i8], [i4, i8, i5],
             [i5, i8, i7], [i5, i7, i6], [i6, i7, i2], [i6, i2, i1],
             [i1, i4, i5], [i1, i5, i6], [i2, i7, i8], [i2, i8, i3]]

    return pymesh.form_mesh(vertices.T, np.array(faces))
