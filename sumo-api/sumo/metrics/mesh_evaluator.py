"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Algorithm class: Evaluate a mesh track submission
"""

from sumo.metrics.Evaluator import Evaluator
import sumo.metrics.utils as utils


class MeshEvaluator(Evaluator):
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

        # sample submission and GT and store in points field
        for e in submission.elements + gt.elements:
            e.points = self._sample_element(e, settings.density)
            
        super(MeshEvaluator, self).__init__(submission, ground_truth, settings)

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
        return utils.points_rmsssd(self, self._submission, self._ground_truth, settings.mesh_overlap_thresh)
        
        
    def rms_color_error(self):
        """
        Compute RMS symmetric surface color distance (RMSSSCD).
          Equation 13 in SUMO white paper.

        Return:
        float - RMSSSCD 
        
        Reference:
        https://www.cs.ox.ac.uk/files/7732/CS-RR-15-08.pdf
        """
        return utils.color_rmsssd(self, submission, ground_truth, settings.voxel_overlap_thresh)


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
        float - mesh IoU (Equation 3 in SUMO white paper)
        """
        return utils.points_iou(element1.points, element2.points, self._config.mesh_overlap_thresh)

    
    def _sample_element(self, element, density=625):
        """
        Sample the faces of <element> at given density.  Uses uniform random sampling.

        Inputs:
        element (ProjectObject) - target element
        density (float) - Number of points per square meter to sample.
          Default 625 gives one point every 4 cm on average.

        Return:
        points (N x 6 np array) - N points by [x,y,z,R,G,B]
        """

        # count the # of faces in all the component meshes
        # and allocate space for faces
        counts = [m.numIndices() for m in element.meshes.primitive_meshes()
                  if m.is_textured()]
        n_faces3 = np.sum(counts)  # 3 x num faces
        faces = np.zeros(n_faces3, 6)  
        
        # build faces list from element component meshes
        start = 0
        for i, mesh in enumerate(element.meshes.primitive_meshes()):
            if mesh.is_textured():
                temp = mesh.vertices()[:, mesh.indices()]
                faces[start:start+counts[i],0:3] = temp.T

                h, w = base_color_texture.shape[0:2]
                
                r = np.mod(np.floor(mesh.uv_coords()[0,:] * h))
                c = np.mod(np.floor(mesh.uv_coords()[1,:] * w))
                
                colors = base_color_texture[r, c, :]
                faces[start:start+counts[i], 3:6] = colors
                
            else:
                print("Warning: skipping sampling of non-textured mesh.")
            
        # sample 
        return self._sample_mesh(faces, density)
        
    def _sample_mesh(self, faces, density=625):
        """ 
        Sample points from a mesh surface using barycentric coordinates.
        
        Inputs:
        faces (np array - 3*N x D) -  matrix representing vertices and faces with
          [X, Y, Z, ...].  faces[0:3, :] is the first face. N is the
          number of faces.  D >=3 (columns beyond 3 are interpolated, too)
        density (float) - Number of points per square meter to sample.
          Default 625 gives one point every 4 cm on average.

        Return:
        samples (np array - N X D matrix of sampled points
        """
        A, B, C = faces[0::3, :], faces[1::3, :], faces[2::3, :]
        cross = np.cross(A[:, 0:3] - C[:, 0:3] , B[:, 0:3] - C[:, 0:3])
        areas = 0.5*(np.sqrt(np.sum(cross**2, axis=1)))

        # ::: set minimum of 1 sample?  ::: this will fail if all faces
        # are small enough
        Nsamples_per_face = (density*areas).astype(int)
        N = np.sum(Nsamples_per_face)  # N = total # of samples

        #TODO: Remove after mesh reading bug is fixed :::? what?
        if N == 0:
            return np.empty((0, 3))

        face_ids = np.zeros((N,), dtype=int)  # reserve space for result

        # store indices for each sample (replicating if there are more
        # than 1 sample in a face
        count = 0
        for i, n in enumerate(Nsamples_per_face):
            face_ids[count:count + Nsamples_per_face[i]] = i
            count += Nsamples_per_face[i]

        # compute barycentric coordinates for each sample
        A = A[face_ids, :]; B = B[face_ids, :]; C = C[face_ids, :]
        r = np.random.uniform(0, 1, (N, 2))
        sqrt_r1 = np.sqrt(r[:, 0:1])
        samples = (1 - sqrt_r1)*A + sqrt_r1*(1 - r[:, 1:])*B + sqrt_r1*r[:, 1:]*C
        return samples
    
