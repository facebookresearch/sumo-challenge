"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Algorithm class: Evaluate a voxel track submission
"""

from sumo.metrics.evaluator import Evaluator
from sumo.metrics.mesh_evaluator import sample_element
import sumo.metrics.utils as utils

class VoxelEvaluator(Evaluator):
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
        settings (EasyDict) - configuration for the evaluator.  See
        Evaluator.py for recognized keys and values.   
        """


        # TODO: not needed anymore.  Use this as a template for converter
        # TODO: Add check that scene type is voxels
        
        # # sample GT, voxelize, and store in voxels field
        # for e in ground_truth.elements:
        #     points = sample_element(e, settings.density)
        #     bbox = ComputeBbox.from_point_cloud(points[0:3].T)
        #     e.voxels = VoxelGrid(voxel_size, bbox.min_corner, points)

        # extract voxel centers
        for e in submission.elements.values():
            centers = e.voxels.voxel_centers()
            centers_posed = e.pose.transform_all_from(centers.T).T
            e.voxel_centers = centers_posed
        for e in ground_truth.elements.values():
            centers = e.voxels.voxel_centers()
            centers_posed = e.pose.transform_all_from(centers.T).T
            e.voxel_centers = centers_posed
            
        super(VoxelEvaluator, self).__init__(ground_truth, submission, settings)


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
        return utils.points_rmsssd(self, self._submission, self._ground_truth,
                                   self._settings.voxel_overlap_thresh, voxels=True)
        
        
    def rms_color_error(self):
        """
        Compute RMS symmetric surface color distance (RMSSSCD).
          Equation 13 in SUMO white paper.

        Return:
        float - RMSSSCD 
        
        Reference:
        https://www.cs.ox.ac.uk/files/7732/CS-RR-15-08.pdf
        """
        return utils.color_rmsssd(self, self._submission, self._ground_truth,
                                  self._settings.voxel_overlap_thresh, voxels=True)
    

    def perceptual_score(self):
        """
            Computes perceptual score for a participant's submission
            Returns:
                perceptual_score: a list of 3 tuples [(s1, s1m), (s2, s2m),
                (s3, s3m)] where s1, s2, s3 are layout, furniture and clutter
                scores respectively and s1m, s2m, s3m are the maximum possible
                scores for layout, furniture, and clutter respectively.
        """
        print("Warning: perceptual_score not implemented yet")
        return 0
    
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
        sim = utils.points_iou(element1.voxel_centers, element2.voxel_centers,
                               self._settings.voxel_overlap_thresh)
        return sim
