"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np

class ModelSampler(object):
    """
    Algorithm to sample points on a GltfModel.
    """

    def __init__(self, sampling_density=625):
        self._sampling_density = sampling_density

    def run(self, model):
        """
        Randomly sample points from the faces of <model> and return
        the resulting points.  Colors are interpolated from the vertex
        colors of the sampled face.

        Inputs:
        model (GltfModel) - model to sample

        Return:
        points (N x 6 numpy array) - N points by 6 (x,y,z,R,G,B)
        """

        # count the # of faces in all the component meshes
        # and allocate space for faces
        counts = [m.num_indices() for m in model.primitive_meshes()]
        n_faces3 = np.sum(counts)  # 3 x num faces
        faces = np.zeros((n_faces3, 6))

        # build faces list from element component meshes
        start = 0

        for i, mesh in enumerate(model.primitive_meshes()):
            # replicate verts based on face ids
            # Note: temp[:,i] = vertex for face_id i
            temp = mesh.vertices()[:, mesh.indices()]  # 3 x N_indices
            faces[start:start+counts[i],0:3] = temp.T  # faces[i,:] = vertex

            h, w = 0, 0  #  texture height and width
            if mesh.is_textured():
                h, w = mesh.base_color().shape[0:2]

            if not mesh.is_textured() or h == 0 or w == 0:
                # base color is not defined (some bad models have this problem)
                # In this case, we just use [128,128,128] (grey) for the color.
                colors = np.array([[128, 128, 128]])
            else:
                uv_coords = mesh.uv_coords()  # 2 x N_verts
                temp2 = uv_coords[:, mesh.indices()]  # 2 x N_indices
                r = np.mod(np.floor(temp2[0,:] * h).astype(int), h)  # 1 x n_vertices
                c = np.mod(np.floor(temp2[1,:] * w).astype(int), w)  # "
                colors = mesh.base_color()[r, c, :]

            faces[start:start+counts[i], 3:6] = colors
            start += counts[i]

        # sample 
        return _sample_mesh(faces, self._sampling_density)
        

# ------------------------------------
# Helper functions
# -------------------------------------
    
def _sample_mesh(faces, density=625):
    """ 
    Sample points from a mesh surface.
    
    Inputs:
    faces (np array - 3*N x D) -  matrix representing vertices and faces with
    [X, Y, Z, ...].  faces[0:3, :] is the first face. N is the
    number of faces.  D >=3 (columns beyond 3 are interpolated, too)
    density (float) - Number of points per square meter to sample.
    Default 625 gives one point every 4 cm on average.

    Return:
    samples (np array - N X D matrix of sampled points

    Algorithm: 
    1. Compute number of samples per face (N_i) based on individual triangle
    areas and target density (points/unit-area).  Small faces are
    guaranteed to contain at least one sample, which may lead to
    higher than the target density for meshes with many small faces.
    2. Randomly sample N_i points from face i using random values in
    barycentric coordinates.
    """

    # Compute triangle areas
    A, B, C = faces[0::3, :], faces[1::3, :], faces[2::3, :]
    cross = np.cross(A[:, 0:3] - C[:, 0:3] , B[:, 0:3] - C[:, 0:3])
    areas = 0.5 * (np.sqrt(np.sum(cross**2, axis=1)))

    # Compute # of samples per face (at least 1 sample per face)
    Nsamples_per_face = np.maximum(1, (density*areas)).astype(int)
    N = np.sum(Nsamples_per_face)  # N = total # of samples

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
        
