#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Algorithm class: Evaluate a mesh track submission
"""

from copy import deepcopy
import math
import numpy as np
from scipy.linalg import logm

from sumo.geometry.rot3 import Rot3
import sumo.metrics.utils as utils
from sumo.semantic.project_object_dict import ProjectObjectDict
from sumo.semantic.object_symmetry import SymmetryType


class Evaluator():
    """
    Base class for evaluating a submission.

    Do not instantiate objects of this class.  Instead, use one of the
    track-specific sub-classes.

    Configuration:
    The algorithm is configured using the settings dict.  Recognized keys:
    thresholds (numpy vector of float) - values for IoU thresholds (tau) at
      which the similarity will be measured.  Default 0.5 to 0.95 in
      0.05 increments.  Thresholds will be sorted in increasing order.
    recall_samples (numpy vector of float) - recall values where PR
      curve will be sampled for average precision computation.
      Default 0 to 1 in 0.01 increments.
    categories (list of string) - categories for which the semantic metric
      should be evaluated.  Default: a small subset of the SUMO
      evaluation categories (see default_settings)
    density (float) - sampling density (in points/square meter).  Used
      in meshes track only.
    mesh_overlap_thresh (float) - max distance for establishing
      a correspondence between two points sampled from mesh surfaces (meters)
    """

    def __init__(self, submission, ground_truth, settings=None):
        """
        Constructor.

        Computes and caches shape similarity and data association
        computations.

        Inputs:
        submission (ProjectScene) - Submitted scene to be evaluated
        ground_truth (ProjectScene) - The ground truth scene
        settings (dict) - configuration for the evaluator.  See top of
          file for recognized attributes and values.
        """
        self._settings = self.default_settings() if settings is None else settings
        self._settings["thresholds"] = np.sort(self._settings["thresholds"])
        self._submission = submission
        self._ground_truth = ground_truth

        # compute similarity between all detections and gt elements
        self._similarity_cache = self._make_similarity_cache(submission, ground_truth)

        # compute-agnostic and category-aware association (for all thresholds)
        self._agnostic_data_assoc = self._compute_agnostic_data_assoc(
            submission.elements, ground_truth.elements,
            self._settings["thresholds"], self._similarity_cache
        )

        self._category_data_assoc = self._compute_category_data_assoc(
            submission.elements, ground_truth.elements,
            self._settings["thresholds"],
            self._settings["categories"], self._similarity_cache
        )

    @staticmethod
    def default_settings():
        """
        Create and return an dict containing default settings.
        """

        thresholds = np.linspace(0.5, 0.95, 10)
        recall_samples = np.linspace(0, 1, 101)
        categories = ["wall", "chair"]
        density = 625
        mesh_overlap_thresh = 0.1  # meters
        voxel_overlap_thresh = 0.1  # meters

        return {
            "thresholds": thresholds,
            "recall_samples": recall_samples,
            "categories": categories,
            "density": density,
            "mesh_overlap_thresh": mesh_overlap_thresh,
            "voxel_overlap_thresh": voxel_overlap_thresh, }

    def evaluate_all(self):
        """
        Computes all metrics for the submission.

        Return:
        dict with key: value pairs (keys are strings - values are corresponding
        evaluation metrics.  Exact keys depend on evaluation track.
        """
        raise NotImplementedError('Instantiate a child class')

    def shape_score(self):
        """
        Compute shape similarity score (Equation 6 in SUMO white paper)

        Return:
        float - shape similarity score
        """

        n_gt = len(self._ground_truth.elements)

        aps = []  # average precision list
        for t in self._settings["thresholds"]:

            # construct input needed for PR curve computation
            # det_matches = 1 if correct detection, 0 if false positive
            det_matches = []
            det_scores = []
            for element in self._submission.elements.values():
                if element.id in self._agnostic_data_assoc[t]:
                    det_matches.append(1)  # correct detection
                else:
                    det_matches.append(0)  # false positive
                det_scores.append(element.score)

            (precision, _) = utils.compute_pr(
                det_matches=np.array(det_matches),
                det_scores=np.array(det_scores),
                n_gt=n_gt,
                recall_samples=self._settings["recall_samples"],
                interp=True)

            aps.append(np.mean(precision))  # Equation 4
        return np.mean(aps)   # Equation 6

    def pose_error(self):
        """
        Compute pose error for the submission.  The pose error
        consists of a rotation error, which is the average geodesic
        distance between detected and ground truth rotations, and a
        translation error, which is the average translation difference
        beween detected and ground truth translations.  See SUMO white paper
        for details.

        Return:
        (rotation_error, translation_error) - where
        rotation_error (float) - Equation 7 in SUMO white paper
        translation_error (float) - Equation 9 in sumo paper

        Note: If submission has no correspondences with any ground truth objects,
        return value is (None, None) and the error is not defined.
        """

        rot_errors = []
        trans_errors = []

        for t in self._settings["thresholds"]:
            rot_errors1, trans_errors1 = [], []
            for corr in self._agnostic_data_assoc[t].values():
                det_element = self._submission.elements[corr.det_id]
                gt_element = self._ground_truth.elements[corr.gt_id]

                #  Eq. 8
                rot_errors1.append(self.rotation_error_1(det_element, gt_element))
                # Eq. 10
                trans_errors1.append(np.linalg.norm(
                    gt_element.pose.t - det_element.pose.t))

            if len(rot_errors1) > 0:
                rot_errors.append(np.mean(rot_errors1))
                trans_errors.append(np.mean(trans_errors1))

        # Eqs. 7 and 9
        if len(rot_errors) == 0:
            return (None, None)
        else:
            return np.mean(rot_errors), np.mean(trans_errors)

    def rotation_error_1(self, det_element, gt_element):
        """
        Compute rotation error between <det_element> and <gt_element>.

        Inputs:
        det_element (ProjectObject) - detected element
        gt_element (ProjectObject) - ground truth element

        Return:
        float - rotation error metric (Eq. 8)
        """
        sym_list = [gt_element.symmetry.x_symmetry,
                    gt_element.symmetry.y_symmetry,
                    gt_element.symmetry.z_symmetry]

        # collect a list of equvalent rots for the target object in rots_to_check
        rots_to_check = [gt_element.pose.R]
        for (axis, sym) in enumerate(sym_list):
            if sym == SymmetryType.spherical:
                # spherical means 0 error
                return 0
            elif sym == SymmetryType.twoFold:
                rots_to_check.extend(
                    [self.symmetric_rot(rot, axis, math.pi) for rot in rots_to_check])
            elif sym == SymmetryType.fourFold:
                new_rots = [self.symmetric_rot(rot, axis, 0.5 * math.pi)
                    for rot in rots_to_check]
                new_rots.extend([self.symmetric_rot(rot, axis, math.pi)
                    for rot in rots_to_check])
                new_rots.extend([self.symmetric_rot(rot, axis, 1.5 * math.pi)
                    for rot in rots_to_check])
                rots_to_check.extend(new_rots)
            elif sym == SymmetryType.cylindrical:
                rots_to_check = [
                    self.cylindrical_symmetric_rot(det_element.pose.R, rot, axis)
                    for rot in rots_to_check]
            # else sym = SymmetryType.none -> do nothing

        # get minimum error
        # Eq. 8, source: https://chrischoy.github.io/research/measuring-rotation/
        errors = [np.sqrt(0.5) *
                  np.linalg.norm(logm(np.matmul(det_element.pose.R.R.T, gt_rot.R)))
                  for gt_rot in rots_to_check]
        return min(errors)

    def symmetric_rot(self, rot, axis, rotation):
        """
        Pre-rotate <rot> by <rotation> radians about <axis>.

        Inputs:
        rot (Rot3) - original rotation
        axis (int) - rotation axis to use (0=x, 1=y, 2=z)
        rotation (float) - rotation amount (radians)
        """
        axis_rot_func = [Rot3.Rx, Rot3.Ry, Rot3.Rz]
        return rot * axis_rot_func[axis](rotation)

    def cylindrical_symmetric_rot(self, det_rot, gt_rot, axis):
        """
        Compute rot that zeroes out the rotation error between
        <det_rot> and <gt_rot> along axis <axis>.

        Inputs
        det_rot (Rot3) - rotation for detection
        gt_rot (Rot3) - rotation for ground truth
        axis (int) - cylindrical rotation axis (0=x, 1=y, 2=z)

        Return:
        Rot3 - new rotation matrix with gt set to det rotation on target axis

        Algorithm:
        1. convert to zyx Euler angles
        2. set gt to detection value for target axis
        3. convert back to matrix rep
        """
        det_euler = utils.matrix_to_euler(det_rot.R)
        gt_euler = utils.matrix_to_euler(gt_rot.R)
        # note e[0] is z
        det_euler[2 - axis] = gt_euler[2 - axis]
        return Rot3(utils.euler_to_matrix(det_euler))

    def rms_points_error(self):
        """
        Compute RMS symmetric surface distance (RMSSSD).
        Equation 11 in SUMO white paper.  This is only defined for
        voxels and mesh tracks.

        Return:
        float - RMSSSD

        Reference:
        https://www.cs.ox.ac.uk/files/7732/CS-RR-15-08.pdf
        """
        raise NotImplementedError('Instantiate a child class')

    def rms_color_error(self):
        """
        Compute RMS symmetric surface color distance (RMSSSCD).
        Equation 13 in SUMO white paper.  This is only defined for
        voxels and mesh tracks.

        Return:
        float - RMSSSCD

        Reference:
        https://www.cs.ox.ac.uk/files/7732/CS-RR-15-08.pdf
        """
        raise NotImplementedError('Instantiate a child class')

    def semantics_score(self):
        """
        Compute semantic score for the submission.
        Return:
        semantic_score (float) - mean Average Precision (mean AP across categories)
        of submission (Equation 15 in SUMO white paper)
        """

        # Initialize:
        # aps = average precision list (1 per category)
        aps = []

        # compute number of GT elements in each category
        # n_gt[cat] = number of GT elements in that category
        n_gt = {}
        for cat in self._settings["categories"]:
            n_gt[cat] = 0
        for element in self._submission.elements.values():
            if element.category in self._settings["categories"]:
                n_gt[element.category] += 1

        for t in self._settings["thresholds"]:
            # det_matches[cat] = list of matches (1 for each detection).
            det_matches = {}
            #   Entry is 1 for a correct match, 0 for a false positive
            # det_scores[cat] = list of detection scores (1 for each detection).
            det_scores = {}

            # initialize
            for cat in self._settings["categories"]:
                det_matches[cat] = []
                det_scores[cat] = []

            # build lists of matches per category
            for element in self._submission.elements.values():
                cat = element.category
                if cat in self._settings["categories"]:
                    if element.id in self._category_data_assoc[cat][t]:
                        det_matches[cat].append(1)  # correct detection
                    else:
                        det_matches[cat].append(0)  # false positive
                    det_scores[cat].append(element.score)

            # compute PR curve per category
            for c in self._settings["categories"]:
                (precision, _) = utils.compute_pr(
                    det_matches=np.array(det_matches[c]),
                    det_scores=np.array(det_scores[c]),
                    n_gt=n_gt[c],
                    recall_samples=self._settings["recall_samples"],
                    interp=True)
                aps.append(np.mean(precision))  # Equation 15
        return np.mean(aps)  # Equation 16

    def perceptual_score(self):
        """
        ::: need to fix
            Computes perceptual score for a participant's submission
            Returns:
                perceptual_score: a list of 3 tuples [(s1, s1m), (s2, s2m),
                (s3, s3m)] where s1, s2, s3 are layout, furniture and clutter
                scores respectively and s1m, s2m, s3m are the maximum possible
                scores for layout, furniture, and clutter respectively.
        """
        raise NotImplementedError('Instantiate a child class')


#------------------------
# End of public interface
#------------------------

    def _make_similarity_cache(self, submission, ground_truth):
        """
        Compute similarity between each pair of elements in submission
        and ground truth.

        Inputs:
        submission (ProjectScene) - Submitted scene to be evaluated
        ground_truth (ProjectScene) - The ground truth scene

        Return:
        similarity_cache (dict of dict of Corr) -
        similarity_cache[det_id][gt_id] = corr, where corr stores the
        similarity and detection score for a putative match between a
        detection and a ground truth element (det_id & gt_id).
        """

        sim_cache = {}
        for det_element in submission.elements.values():
            det_id = det_element.id
            sim_cache[det_id] = {}
            det_score = det_element.score
            for gt_element in ground_truth.elements.values():
                corr = Corr(det_id=det_id,
                            gt_id=gt_element.id,
                            similarity=self._shape_similarity(det_element, gt_element),
                            det_score=det_score)
                sim_cache[det_id][gt_element.id] = corr
        return sim_cache

    def _shape_similarity(self, element1, element2):
        """
        Similarity function that compares the shape of <element1> and
        <element2>.
        The actual similarity function must be defined in
        track-specific child classes.

        Inputs:
        element1 (ProjectObject)
        element2 (ProjectObject)

        Return:
        float - shape similarity score
        """
        raise NotImplementedError('Instantiate a child class')

    def _compute_agnostic_data_assoc(self, det_elements, gt_elements,
          thresholds, sim_cache):

        """
        Computes agnostic (category-independent) data association
        between the elements in <submission> and <ground_truth> for
        each similarity threshold in <thresholds>.

        Inputs:
        det_elements (ProjectObjectDict) - submitted/detected scene elements
        gt_elements (ProjectObjectDict) - corresponding ground truth
          scene elements
        thresholds (numpy vector of float) - Similarity thresholds to be used.
        sim_cache (dict of dict of Corrs) - similarity cache -
          sim_cache[det_id][gt_id] is similarity between det_id and gt_id.

        Return:
        data_assoc (dict of dicts of Corr) -
        data_assoc[thresh][det_id], where thresh is taken from
        <thresholds> and det_id is a detection ID.  If a det_id is not
        in the dict, no correspondance was found.

        Algorithm:
        1. Matches with similarity < thresh are eliminated from
        consideration.
        2. The remaining detections are sorted by decreasing detection
        score.
        3. Loop over detections.  The detection with highest score is
        assigned to its corresponding GT, and that detection and GT
        are removed from further consideration.  If a detection has
        multiple possible matches in the GT, the match with highest
        similarity score is used.
        """

        # make copy of the cache that we can modify
        sim_cache2 = deepcopy(sim_cache)

        # for storing results
        data_assoc = {}  # key is threshold

        # loop in increasing similarity threshold order
        # Note: This allows us to reuse edits to similarity cache,
        # since any putative matches ruled out for a given threshold
        # will also be ruled out for higher thresholds.
        for thresh in thresholds:
            data_assoc[thresh] = {}

            # remove matches with similarity < thresh
            for det_id in sim_cache2.keys():
                for gt_id in list(sim_cache2[det_id].keys()):
                    if sim_cache2[det_id][gt_id].similarity < thresh:
                        sim_cache2[det_id].pop(gt_id)

            # for tracking GT elements that have already been assigned
            # at this threshold
            assigned_gts = {}  # key is gt_id, val is det_id

            # iterate over detections sorted in descending order of
            # score.
            for det_element in sorted(
                    det_elements.values(),
                    key=lambda element: element.score,
                    reverse=True):
                det_id = det_element.id

                # make list of possible matches for this det_id and
                # sort by similarity
                possible_corrs = [
                    corr for corr in sim_cache2[det_id].values()
                    if corr.gt_id not in assigned_gts]
                sort_corrs_by_similarity(possible_corrs)

                # create match with last corr in list
                if len(possible_corrs) > 0:
                    data_assoc[thresh][det_id] = possible_corrs[-1]
                    assigned_gts[possible_corrs[-1].gt_id] = det_id
                # else no match for this det_id

        return data_assoc

    def _compute_category_data_assoc(self, det_elements, gt_elements,
                                  thresholds, categories, sim_cache):
        """
        Computes category-specific data association
        between the elements in <submission> and <ground_truth> for
        each similarity threshold in <thresholds>.

        Inputs:
        det_elements (ProjectObjectDict) - submitted/detected scene elements
        gt_elements (ProjectObjectDict) - corresponding ground truth
          scene elements
        thresholds (numpy vector of float) - Similarity thresholds to be used.
        sim_cache (dict of dict of Corrs) - similarity cache -
          sim_cache[det_id][gt_id] is similarity between det_id and gt_id.

        Return:
        data_assoc (dict of dicts of dicts of Corr) -
        data_assoc[category][thresh][det_id], where thresh is taken from
        <thresholds> and det_id is a detection ID.  If a det_id is not
        in the dict, it means that no correspondance was found.


        Algorithm:
        For each category C in category list (from settings):
        1. Get a list of elements in submission and GT belonging to
        that category;  If both lists are empty, we skip the category.
        2. Construct submission and ground truth ProjectScenes using
        only elements with category C.
        3. Compute category-agnostic data association using these subsets.

        """

        # Split detections and gt elements by category and store in
        # dict of ProjectObjectDicts.

        dets_by_cat = {}
        gts_by_cat = {}
        for cat in categories:
            dets_by_cat[cat] = ProjectObjectDict()
            gts_by_cat[cat] = ProjectObjectDict()

        for (id, element) in det_elements.items():
            if element.category in categories:
                dets_by_cat[element.category][id] = element

        for (id, element) in gt_elements.items():
            if element.category in categories:
                gts_by_cat[element.category][id] = element

        data_assoc = {}  # for storing results (key is category)

        for cat in categories:
            dets = dets_by_cat[cat]
            gts = gts_by_cat[cat]
            if (len(dets) + len(gts)) == 0:
                continue

            # build mini sim_cache just for this category
            sim_cache_cat = {}
            for det_id in dets.keys():
                sim_cache_cat[det_id] = {}
                for gt_id in gts.keys():
                    if gt_id in sim_cache[det_id]:
                        sim_cache_cat[det_id][gt_id] = sim_cache[det_id][gt_id]

            # do data association
            data_assoc[cat] = self._compute_agnostic_data_assoc(
                dets, gts, thresholds, sim_cache_cat)

        return data_assoc

    def _print_sim_cache(self, sim_cache):
        for det_id in sim_cache.keys():
            for gt_id in sim_cache[det_id].keys():
                corr = sim_cache[det_id][gt_id]
                print("Det: {} | GT: {} | sim: {} | score: {}".format(
                    det_id, gt_id, corr.similarity, corr.det_score))


class Corr():
    """
    Helper class for storing a correspondence.
    """
    __slots__ = 'det_id', 'gt_id', 'similarity', 'det_score'

    def __init__(self, det_id, gt_id, similarity, det_score):
        self.det_id = det_id
        self.gt_id = gt_id
        self.similarity = similarity
        self.det_score = det_score


def sort_corrs_by_similarity(corrs):
    """
    Sort a list of correspondences by similarity.  Sort is in place.

    Inputs:
    corrs (list of Corr) - correspondences to sort.
    """
    corrs.sort(key=lambda corr: corr.similarity)
