#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Algorithm class: Evaluate a mesh track submission
"""
import numpy as np

from sumo.metrics.evaluator import Evaluator
from sumo.threedee.compute_bbox import ComputeBbox
import sumo.metrics.utils as utils

# ------------------------------------
# Helper functions (not part of class)
# -------------------------------------


def sample_element(element, density=625):
    """
    Sample the faces of <element> at given density.  Uses uniform
    random sampling.  ***Note***: The resulting points are in room coordinates.

    Inputs:
    element (ProjectObject) - target element
    density (float) - Number of points per square meter to sample.
      Default 625 gives one point every 4 cm on average.

    Return:
    points (N x 6 np array) - N points by [x,y,z,R,G,B]
    """

    # count the # of faces in all the component meshes
    # and allocate space for faces
    counts = [m.num_indices() for m in element.meshes.primitive_meshes()]
    n_faces3 = np.sum(counts)  # 3 x num faces
    faces = np.zeros((n_faces3, 6))

    # build faces list from element component meshes
    start = 0

    for i, mesh in enumerate(element.meshes.primitive_meshes()):
        # transform mesh to room coordinates
        posed_mesh = mesh.transform(element.pose)

        # replicate verts based on face ids
        # temp[:,i] = vertex for face_id i
        temp = posed_mesh.vertices()[:, posed_mesh.indices()]  # 3 x N_indices
        faces[start:start + counts[i], 0:3] = temp.T  # faces[i,:] = vertex

        h, w = 0, 0  # texture height and width
        if posed_mesh.is_textured():
            h, w = posed_mesh.base_color().shape[0:2]

        if not posed_mesh.is_textured() or h == 0 or w == 0:
            # base color is not defined (some bad models have this problem)
            # In this case, we just use [128,128,128] (grey) for the color.
            colors = np.array([[128, 128, 128]])
        else:
            uv_coords = posed_mesh.uv_coords()  # 2 x N_verts
            temp2 = uv_coords[:, posed_mesh.indices()]  # 2 x N_indices
            r = np.mod(np.floor(temp2[0, :] * h).astype(int), h)  # 1 x n_vertices
            c = np.mod(np.floor(temp2[1, :] * w).astype(int), w)  # "
            colors = posed_mesh.base_color()[r, c, :]

        faces[start:start + counts[i], 3:6] = colors
        start += counts[i]

    # sample
    return sample_mesh(faces, density)


def sample_mesh(faces, density=625):
    """
    Sample points from a mesh surface using barycentric coordinates.

    Inputs:
    faces (np array - 3*N x D) -  matrix representing vertices and faces with
    [X, Y, Z, ...].  faces[0:3, :] is the first face. N is the
    number of faces.  D >=3 (columns beyond 3 are interpolated, too)
    density (float) - Number of points per unit square surface to sample.
    Default 625 gives one point every 4 cm on average in metric units.

    Return:
    samples (np array - N X D matrix of sampled points
    """
    A, B, C = faces[0::3, :], faces[1::3, :], faces[2::3, :]
    cross = np.cross(A[:, 0:3] - C[:, 0:3] , B[:, 0:3] - C[:, 0:3])
    areas = 0.5 * (np.sqrt(np.sum(cross**2, axis=1)))

    Nsamples_per_face = (density * areas).astype(int)
    # Set minimum of 1 sample, otherwise for meshes with small
    # faces, there will be no samples.  The downside is that
    # for meshes with small faces, the density will be higher than
    # requested.
    # TODO: Implement probabilistic face sampling weighted by
    # normalized face area.
    Nsamples_per_face = np.clip(Nsamples_per_face, a_min=1, a_max=None)
    N = np.sum(Nsamples_per_face)  # N = total # of samples

    face_ids = np.zeros((N,), dtype=np.int64)  # reserve space for result

    # store indices for each sample (replicating if there are more
    # than 1 sample in a face
    count = 0
    for i, _ in enumerate(Nsamples_per_face):
        face_ids[count:count + Nsamples_per_face[i]] = i
        count += Nsamples_per_face[i]

    # compute barycentric coordinates for each sample
    A = A[face_ids, :]
    B = B[face_ids, :]
    C = C[face_ids, :]
    r = np.random.uniform(0, 1, (N, 2))
    sqrt_r1 = np.sqrt(r[:, 0:1])
    samples = (1 - sqrt_r1) * A + sqrt_r1 * (1 - r[:, 1:]) * B + sqrt_r1 * r[:, 1:] * C
    return samples


class MeshEvaluator(Evaluator):
    """
    Algorithm to evaluate a submission for the mesh track.
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

        # sample submission and GT and store in points field
        # also compute bounding box of each posed object
        for e in submission.elements.values():
            e.posed_points = sample_element(e, settings["density"])
            e.posed_bbox = ComputeBbox().from_point_cloud(e.posed_points[:, 0:3].T)
        for e in ground_truth.elements.values():
            e.posed_points = sample_element(e, settings["density"])
            e.posed_bbox = ComputeBbox().from_point_cloud(e.posed_points[:, 0:3].T)

        super(MeshEvaluator, self).__init__(submission, ground_truth, settings)

    def evaluate_all(self):
        """
        Compute all metrics for the submission

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
        metrics["rms_points_error"] = self.rms_points_error()
        metrics["rms_color_error"] = self.rms_color_error()
        metrics["semantics_score"] = self.semantics_score()
        metrics["perceptual_score"] = self.perceptual_score()

        return metrics

    def rms_points_error(self):
        """
        Compute RMS symmetric surface distance (RMSSSD).
          Equation 11 in SUMO white paper.

        Return:
        float - RMSSSD

        Reference:
        https://www.cs.ox.ac.uk/files/7732/CS-RR-15-08.pdf
        """
        return utils.points_rmsssd(self, self._submission, self._ground_truth)

    def rms_color_error(self):
        """
        Compute RMS symmetric surface color distance (RMSSSCD).
          Equation 13 in SUMO white paper.

        Return:
        float - RMSSSCD

        Reference:
        https://www.cs.ox.ac.uk/files/7732/CS-RR-15-08.pdf
        """
        return utils.color_rmsssd(self, self._submission, self._ground_truth)

#------------------------
# End of public interface
#------------------------

    def _shape_similarity(self, element1, element2):
        """
        Similarity function that compares the mesh overlap of
        <element1> and <element2>.

        Inputs:
        element1 (ProjectObject)
        element2 (ProjectObject)

        Return:
        float - mesh IoU (Equation 3 in SUMO white paper)
        """
        sim = utils.points_iou(element1.posed_points, element1.posed_bbox,
                               element2.posed_points, element2.posed_bbox,
                               self._settings["mesh_overlap_thresh"])
        return sim
