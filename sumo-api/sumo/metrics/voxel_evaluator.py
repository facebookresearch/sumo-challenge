"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Algorithm class: Evaluate a voxel track submission
"""

from sumo.metrics.Evaluator import Evaluator
import sumo.metrics.utils as utils
from sumo.threedee.compute_bbox import ComputeBbox

class VoxelEvaluator(Evaluator):
    """
    Algorithm to evaluate a submission for the mesh track.
    """

    def __init__(self, submission, ground_truth, settings):
        """
        Constructor.  Computes similarity between all elements in the
        submission and ground_truth and also computes amodal and modal
        data association caches. 

        Inputs:
        submission (ProjectScene) - Submitted scene to be evaluated
        ground_truth (ProjectScene) - The ground truth scene
        settings (EasyDict) - configuration for the evaluator.  See
        Evaluator.py for recognized keys and values.   
        """

        
        # sample GT, voxelize, and store in voxels field
        for e in gt.elements:
            points = self._sample_element(e, settings.density)
            bbox = ComputeBbox.from_point_cloud(points[0:3].T)
            e.voxels = VoxelGrid(voxel_size, bbox.min_corner, points)

        # extract voxel centers
        for e in submission.elements + gt.elements:
            e.voxel_centers = e.voxels.voxel_centers()
            
        super(VoxelEvaluator, self).__init__(config, ground_truth, submission)


    def evaluate_all(self)
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

        metrics["shape_score"] = self.shape_similarity_score()
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
        return utils.points_rmsssd(self, self._submission, self._ground_truth, settings.voxel_overlap_thresh, voxels=True)
        
        
    def rms_color_error(self):
        """
        Compute RMS symmetric surface color distance (RMSSSCD).
          Equation 13 in SUMO white paper.

        Return:
        float - RMSSSCD 
        
        Reference:
        https://www.cs.ox.ac.uk/files/7732/CS-RR-15-08.pdf
        """
        return utils.color_rmsssd(self, submission, ground_truth, settings.voxel_overlap_thresh, voxels=True)
    

    def evaluate_perceptual(self, submission, ground_truth):
        """
            Computes perceptual score for a participant's submission
            Args:
                ground_truth: Ground-truth - list of ProjectObject instances
                submission: Submission - list of ProjectObject instances
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

    def _shape_similarity(self, element1, element2):
        """
        Similarity function that compares the mesh overlap of
        <element1> and <element2>.

        Inputs:
        element1 (ProjectObject)
        element2 (ProjectObject) 

        Return:
        float - voxel IoU (Equation 2 in SUMO white paper)
        """
        return utils.points_iou(element1.voxel_centers, element2.voxel_centers, self._config.voxel_overlap_thresh)
        
        

    def evaluate_geometry(self, ground_truth, submission):
        """
            Computes shape score, pose score, and voxel root mean squared
            symmetric distance for a participant's submission
            Args:
                ground_truth: Ground-truth - list of ProjectObject instances
                submission: Submission - list of ProjectObject instances
            Returns:
                pose_score: (R_error, t_error) tuple representing average
                rotation matrix geodesic distance, and translation error
                shape_score: Average Precision (class-agnostic) of submission
                voxel_rmsd: root mean squared symmetric surface distance of
                voxel centers. See https://www.cs.ox.ac.uk/files/7732/CS-RR-15-08.pdf
        """
        shape_score, pose_score = super(
            VoxelEvaluator, self).evaluate_geometry(ground_truth, submission)
        voxel_rmsd = points_rmsd(self, ground_truth, submission, use_voxels=True)

        return shape_score, pose_score, voxel_rmsd

    def evaluate_appearance(self, ground_truth, submission):
        """
            Computes shape score, pose score, and voxel root mean squared
            symmetric distance for a participant's submission
            Args:
                ground_truth: Ground-truth - list of ProjectObject instances
                submission: Submission - list of ProjectObject instances
            Returns:
                pose_score: (R_error, t_error) tuple representing average
                rotation matrix geodesic distance, and translation error
                shape_score: Average Precision (class-agnostic) of submission
                voxel_color_rmsd: root mean squared symmetric surface distance of
                voxel centers. See https://www.cs.ox.ac.uk/files/7732/CS-RR-15-08.pdf
        """

        voxel_color_rmsd = color_rmsd(self, ground_truth, submission, use_voxels=True)

        return voxel_color_rmsd

    def evaluate_perceptual(self, ground_truth, submission):
        """
            Computes perceptual score for a participant's submission
            Args:
                ground_truth: Ground-truth - list of ProjectObject instances
                submission: Submission - list of ProjectObject instances
            Returns:
                perceptual_score: a list of 3 tuples [(s1, s1m), (s2, s2m),
                (s3, s3m)] where s1, s2, s3 are layout, furniture and clutter
                scores respectively and s1m, s2m, s3m are the maximum possible
                scores for layout, furniture, and clutter respectively.
        """
        raise NotImplementedError('Instantiate a child class')

    def evaluate_all(self, ground_truth, submission):
        """
            Computes all metrics for a participant's submission
            Args:
                ground_truth: Ground-truth - list of ProjectObject instances
                submission: Submission - list of ProjectObject instances
            Returns:
                dictionary containing geometry, appearance, semantics, and
                perceptual metrics for submission
        """
        metrics = {}
        shape_score, pose_score, rmsd = self.evaluate_geometry(
            ground_truth, submission)
        rmscd = self.evaluate_appearance(ground_truth, submission)
        mAP = self.evaluate_semantics(ground_truth, submission)
        metrics[self.config.metrics.AP[0]] = shape_score
        metrics[self.config.metrics.Rerror[0]] = pose_score[0]
        metrics[self.config.metrics.Terror[0]] = pose_score[1]
        metrics[self.config.metrics.rmsd[0]] = rmsd
        metrics[self.config.metrics.rmscd[0]] = rmscd
        metrics[self.config.metrics.mAP[0]] = mAP
        return metrics
