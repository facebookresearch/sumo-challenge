#!/usr/bin/env python3
"""
    Metrics helper functions
"""

from functools import wraps
import gc
import math
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import numpy as np
import pymesh
import pyny3d.geoms as pyny
from sklearn.neighbors import BallTree  # for nearest neighbors
import timeit

from sumo.geometry.rot3 import Rot3


# Timing utility
#
def measure_time(f):
    """
    Decorator that measures the time for a given function.

    To time a function, use this form:

    from sumo.metrics.utils import measure_time
    @measure_time
    def function_foo(bar):
        pass

    Source: https://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python
    """
    @wraps(f)
    def _wrapper(*args, **kwargs):
        gcold = gc.isenabled()
        gc.disable()
        start_time = timeit.default_timer()
        try:
            result = f(*args, **kwargs)
        finally:
            elapsed = timeit.default_timer() - start_time
            if gcold:
                gc.enable()
            print('Function "{}": {}s'.format(f.__name__, elapsed))
        return result
    return _wrapper


#-------------------------------------------------
# Transformations between rotation representations
#-------------------------------------------------

def matrix_to_quat(M):
    """
    Transform a 3X3 Rotation matrix to a unit quaternion.

    Inputs:
    M (numpy 3x3 array of float) - rotation matrix

    Return:
    q (numpy vector of float) -  quaternion in (w,i,j,k) order

    Source:
    http://www.euclideanspace.com/maths/geometry/rotations/\
    conversions/matrixToQuaternion/\
    www.ee.ucr.edu/~farrell/AidedNavigation/D_App_Quaternions/Rot2Quat.pdf
    """
    m00, m01, m02 = M[0, :]
    m10, m11, m12 = M[1, :]
    m20, m21, m22 = M[2, :]
    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    D = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    return np.array([qw / D, qx / D, qy / D, qz / D])


def quat_to_matrix(quat):
    """
    Convert quaternion to rotation matrix

    Inputs:
    quat (numpy vector of float) - quaternion in (w,i,j,k) order

    Return
    M (numpy 3x3 array of float) - rotation matrix

    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """

    q = math.sqrt(2.0) * quat
    q = np.outer(q, q)
    return np.array([[1 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
                     [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
                     [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]])


def quat_to_euler(q):
    """
    Convert quaternion to static ZYX Euler angle representation
    (i.e., R = R(z)*R(y)*R(x))

    Inputs:
    q (numpy vector of float) - quaternion in (w,i,j,k) order

    Return:
    numpy vector of float - Euler angles in radians (z, y, x)

    Source: matlab quat2eul
    eul = [ atan2( 2*(qx.*qy+qw.*qz), qw.^2 + qx.^2 - qy.^2 - qz.^2 ), ...
    asin( -2*(qx.*qz-qw.*qy) ), ...
    atan2( 2*(qy.*qz+qw.*qx), qw.^2 - qx.^2 - qy.^2 + qz.^2 )];

    """
    return np.array([math.atan2(2 * (q[1] * q[2] + q[0] * q[3]),
                                q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]),
                     math.asin(-2 * (q[1] * q[3] - q[0] * q[2])),
                     math.atan2(2 * (q[2] * q[3] + q[0] * q[1]),
                                q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3])])


def euler_to_quat(e):
    """
    Convert ZYX euler angles to quaternion

    Inputs:
    e (numpy vector of float) - Euler angles in radians (z, y, x)

    Return:
    q (numpy vector of float) - quaternion in (w,i,j,k) order
    """
    cz = math.cos(0.5 * e[0])
    sz = math.sin(0.5 * e[0])
    cy = math.cos(0.5 * e[1])
    sy = math.sin(0.5 * e[1])
    cx = math.cos(0.5 * e[2])
    sx = math.sin(0.5 * e[2])

    return np.array([
        cz * cy * cx + sz * sy * sx,
        cz * cy * sx - sz * sy * cx,
        cz * sy * cx + sz * cy * sx,
        sz * cy * cx - cz * sy * sx])


def euler_to_matrix(e):
    """
    Convert ZYX Euler angles to 3x3 rotation matrix.

    Inputs:
    e (numpy 3-vector of float) - ZYX Euler angles (radians)

    Return:
    matrix (3x3 numpy array2 of float) - rotation matrix

    TODO: This could be optimized somewhat by using the direct
    equations for the final matrix rather than multiplying out the
    matrices.
    """
    return (Rot3.Rz(e[0]) * Rot3.Ry(e[1]) * Rot3.Rx(e[2])).R


def matrix_to_euler(matrix):
    """
    Convert 3x3 matrix to ZYX Euler angles.

    Inputs:
    matrix (numpy 3x3 numpy array2 of float) - rotation matrix

    Return:
    numpy 3-vector of float - ZYX Euler angles (radians)

    TODO:
    This could be written more efficiently going directly between
    matrix and Euler angles, but there are singularities and issues
    with numerical stability to be considered.
    """
    return quat_to_euler(matrix_to_quat(matrix))

#-------------------------------------
# Precision recall curves
#-------------------------------------


def compute_ap(det_matches, det_scores, n_gt, recall_samples=None,
               interp=False, area_under_curve=True):
    """
    Compute average precision and precision-recall curve

    Inputs:
        det_matches (numpy vector of N ints) - Each non-zero entry is a
        correct detection.  Zeroes are false positives.
        det_scores (numpy vector of N floats) - The detection scores for
        the corresponding matches.  Higher is better.
        n_gt (int) - The number of ground truth entities in the task.
        recall_samples (numpy vector of float) - If set, compute precision at
        these sample locations.  Values must be between 0 and 1
        inclusive.
        interp (Boolean) - If true, the interpolated PR curve will be
        generated (as described in :::cite pascal voc paper)
        area_under_curve (Boolean): If True, compute average precision as area
        under the curve
    Return:
        (ap, precision, recall) where
          ap (float) - average precision
          precision (numpy vector of float) - precision values at corresponding <recall> points
          recall (numpy vector of float) - recall values.
    """
    if area_under_curve:
        ap, precision, recall = compute_auc_ap(det_matches, det_scores, n_gt)
    else:
        precision, recall = compute_pr(
            det_matches, det_scores, n_gt, recall_samples, interp)
        ap = np.mean(precision)

    return ap, precision, recall


def compute_pr(det_matches, det_scores, n_gt, recall_samples=None, interp=False):
    """
    Compute the precision-recall curve.

    Inputs:
    det_matches (numpy vector of N ints) - Each non-zero entry is a
      correct detection.  Zeroes are false positives.
      det_scores (numpy vector of N floats) - The detection scores for
      the corresponding matches.  Higher is better.
    n_gt (int) - The number of ground truth entities in the task.
    recall_samples (numpy vector of float) - If set, compute precision at
      these sample locations.  Values must be between 0 and 1
      inclusive.
    interp (Boolean) - If true, the interpolated PR curve will be
      generated (as described in :::cite pascal voc paper)

    Return:
    (precision, recall)
    precision (numpy vector of float) - precision values at corresponding
      <recall> points recall (numpy vector of float) - recall locations.  If
      <recall_samples> is not set, it is the locations where precision changes.
      Otherwise it is set to <recall_samples>.
    """

    # sort input based on score
    indices = np.argsort(-det_scores)
    sorted_matches = det_matches[indices]

    # split out true positives and false positives
    tps = np.not_equal(sorted_matches, 0)
    fps = np.equal(sorted_matches, 0)

    # compute basic PR curve
    tp_sum = np.cumsum(tps)
    fp_sum = np.cumsum(fps)

    # use epsilon to prevent divide by 0 special case
    epsilon = np.spacing(1)

    precision = tp_sum / (tp_sum + fp_sum + epsilon)
    recall = tp_sum / n_gt

    # compute interpolated PR curve
    if (interp):
        for i in range(len(precision) - 1, 0, -1):
            if precision[i] > precision[i - 1]:
                precision[i - 1] = precision[i]

    # compute at recall sample points
    # Note: This is what MS Coco does.  Not sure if it is correct,
    # but it should be sufficient if the number of samples used to
    # create the PR curve is large enough.
    # This assigns the precision value for a given recall_sample to
    # the nearest value on the right.  Anything greater than the last
    # computed recall value will be set to zero.
    if recall_samples is not None:
        n_precision = len(precision)
        precision2 = np.zeros(len(recall_samples))  # default is 0

        indices2 = np.searchsorted(recall, recall_samples, side='left')
        for recall_index, precision_index in enumerate(indices2):
            if (precision_index < n_precision):
                precision2[recall_index] = precision[precision_index]
        precision = precision2
        recall = recall_samples

    return (precision, recall)


def compute_auc_ap(det_matches, det_scores, n_gt):
    """
    Compute average precision as area under the precision-recall curve.

    Inputs:
    det_matches (numpy vector of N ints) - Each non-zero entry is a
      correct detection.  Zeroes are false positives.
      det_scores (numpy vector of N floats) - The detection scores for
      the corresponding matches.  Higher is better.
    n_gt (int) - The number of ground truth entities in the task.
    Return:
        average_precision(float) - area under the PR curve
        precision (numpy vector of float) - precision values at corresponding
          <recall> points
        recall (numpy vector of float) - recall values.
    """
    # sort input based on score
    indices = np.argsort(-det_scores)
    sorted_matches = det_matches[indices]

    # split out true positives and false positives
    tps = np.not_equal(sorted_matches, 0)
    fps = np.equal(sorted_matches, 0)

    # compute basic PR curve
    tp_sum = np.cumsum(tps)
    fp_sum = np.cumsum(fps)

    # use epsilon to prevent divide by 0 special case
    epsilon = np.spacing(1)

    precision = tp_sum / (tp_sum + fp_sum + epsilon)
    recall = tp_sum / n_gt

    ap = 0
    # compute interpolated PR curve and average precision
    if len(precision) > 0:
        for i in range(len(precision) - 1, 0, -1):
            if precision[i] > precision[i - 1]:
                precision[i - 1] = precision[i]
            ap += precision[i] * (recall[i] - recall[i - 1])
        ap += precision[0] * recall[0]

    return ap, precision, recall


def plot_pr(precision, recall):
    """
    Creates a new figure and generates a plot of a precision recall curve.

    Inputs:
    precision (numpy vector of N floats) - precision values at corresponding recall
    recall (numpy vector of N floats) - recall values

    Return:
    Figure - matplotlib Figure object for the plot.

    Notes:
    does not call plt.show()
    """
    fig = plt.figure()
    plt.plot(recall, precision, 'r-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0, 1, 0, 1.1])

    return fig


#----------------------------------------
# Surface and voxel similarity support
#----------------------------------------

def points_iou(points1, bbox1, points2, bbox2, thresh):
    """
    Compute the IoU metric for two point sets.  Equations 2 and 3 in
    SUMO white paper.  The metric is the ratio of overlapping points
    to the total number of points in both sets.  A point is
    overlapping if it's nearest neighbor in the other set is less than
    a threshold distance.

    Inputs:
    points1 (np array N x 3 of float) - points in set 1. N points by 3 coordinates.
    bbox1 (Box3d) - bounding box for points1
    points2 (np array N x 3 of float) - points in set 2. N points by 3 coordinates.
    bbox2 (Box3d) - bounding box for points2
    thresh (float) - threshold for a pair of corresponding points to
      be considered overlapping.

    Return:
    IoU (float) - intersection over union as defined in Eq. 2 and 3 in
    the SUMO white paper.
    """

    # quick intersection test.  If bounding boxes don't overlap on any single axis,
    # then the enclosed object cannot overlap
    for axis in range(3):
        if (bbox1.min_corner[axis] > bbox2.max_corner[axis] + thresh) or \
           (bbox2.min_corner[axis] > bbox1.max_corner[axis] + thresh):
            return 0

    ind1to2, ind2to1, dist1to2, dist2to1 = nearest_neighbors(
        points1[:, 0:3], points2[:, 0:3])
    intersection = np.sum(dist1to2 <= thresh) + np.sum(dist2to1 <= thresh)
    union = points1.shape[0] + points2.shape[0]
    if union == 0:
        return 0
    else:
        return intersection / union


def nearest_neighbors(points1, points2):
    """
    Compute bidirectional nearest neighbors between two sets of points.

    Inputs:
    points1 (np array N x 3 of float) - points in set 1. N points by 3 coordinates.
    points2 (np array N x 3 of float) - points in set 2. N points by 3 coordinates.

    Return:
    (ind1to2, ind2to1, dist1to2, dist2to1) - tuple where:
    ind1to2 (np vector (N,) of int) - closest points for points1
      (ind1to2[i] = index from points2 that is closest to points1[i,:])
    ind2to1 (np vector (N,) of int) - closest points for points2
      (ind2to1[i] = index from points1 that is closest to points2[i,:])
    dist1to2 (np vector (N,) of float) - distances from points1 to closest
      points in points2 (dist1to2[i] = distance between points1[i,:] and
      points2[ind1to2[i],:])
    dist2to1 (np vector (N,) of float) - distances from points2 to closest
      points in points1 (dist2to1[i] = distance between points2[i,:] and
      points1[ind2to1[i],:])
    """

    # Note: There is a bug/limitation in sklearn that it cannot handle a query
    # with no points.  This is the workaround...
    if points1.shape[0] == 0 or points2.shape[0] == 0:
        ind1to2 = np.empty(shape=(0,), dtype=np.int64)
        ind2to1 = np.empty(shape=(0,), dtype=np.int64)
        dist1to2 = np.empty(shape=(0,), dtype=np.int64)
        dist2to1 = np.empty(shape=(0,), dtype=np.int64)
    else:
        tree1 = BallTree(points1)
        tree2 = BallTree(points2)
        dist1to2, ind1to2 = tree2.query(points1)
        dist2to1, ind2to1 = tree1.query(points2)
        ind1to2 = ind1to2.flatten()
        ind2to1 = ind2to1.flatten()
    return ind1to2, ind2to1, dist1to2, dist2to1


def points_rmsssd(evaluator, submission, ground_truth, voxels=False):
    """
    Compute average root mean squared symmetric surface distance
    (RMSSSD). Equation 11 in SUMO white paper.

    Inputs:
    submission (ProjectScene) - Submitted scene to be evaluated
    ground_truth (ProjectScene) - The ground truth scene

    Return:
        RMSSSD (float or math.inf if there are no corresponding points
        within the threshold)

    References:
        [2]https://www.cs.ox.ac.uk/files/7732/CS-RR-15-08.pdf
    """

    # cache distances:  rmsssd_cache[dt_id][gt_id] = rmsssd
    rmsssd_cache = {}

    settings = evaluator._settings
    data_assoc = evaluator._agnostic_data_assoc

    rmsssd1 = []  # list of rmsssd per correspondence
    for t in settings["thresholds"]:
        for det_id in data_assoc[t]:
            # skip if not evaluated
            if data_assoc[t][det_id].evaluated == False:
                continue

            if det_id not in rmsssd_cache:
                rmsssd_cache[det_id] = {}
            gt_id = data_assoc[t][det_id].gt_id

            if gt_id in rmsssd_cache[det_id]:
                # get from cache
                rmsssd1.append(rmsssd_cache[det_id][gt_id])
            else:  # compute from scratch
                # TODO: The nearest neighbors are also computed when making
                # the sim_cache.  It should be possible to only do
                # this computation once, but it will require a
                # fair amount of refactoring

                points1 = evaluator._submission.elements[det_id].posed_points
                points2 = evaluator._ground_truth.elements[gt_id].posed_points

                ind1to2, ind2to1, dist1to2, dist2to1 = nearest_neighbors(
                    points1[:, 0:3], points2[:, 0:3])

                n_matched = dist1to2.shape[0] + dist2to1.shape[0]
                if n_matched > 0:
                    # SUMO white paper Eq 12
                    rmsssd = np.sqrt((np.sum(np.square(dist1to2)) +
                                      np.sum(np.square(dist2to1))) / n_matched)
                else:
                    rmsssd = 0
                rmsssd_cache[det_id][gt_id] = rmsssd
                rmsssd1.append(rmsssd)

    if len(rmsssd1) > 0:
        return np.mean(rmsssd1)
    else:
        return math.inf  # no corrs found


def color_rmsssd(evaluator, submission, ground_truth, voxels=False):
    """
    Compute average root mean squared symmetric surface color distance
    (RMSSSCD). Equation 13 in SUMO white paper.

    Inputs:
    submission (ProjectScene) - Submitted scene to be evaluated
    ground_truth (ProjectScene) - The ground truth scene

    Return:
        RMSSSCD (float or math.inf if there are no corresponding points
        within the threshold)

    References:
        [2]https://www.cs.ox.ac.uk/files/7732/CS-RR-15-08.pdf
    """
    # cache distances:  rmssscd_cache[dt_id][gt_id] = rmssscd
    rmssscd_cache = {}

    settings = evaluator._settings
    data_assoc = evaluator._agnostic_data_assoc

    rmssscd1 = []  # list of rmssscd per correspondence
    for t in settings["thresholds"]:
        for det_id in data_assoc[t]:
            # skip if not evaluated
            if data_assoc[t][det_id].evaluated == False:
                continue

            if det_id not in rmssscd_cache:
                rmssscd_cache[det_id] = {}
            gt_id = data_assoc[t][det_id].gt_id
            if gt_id in rmssscd_cache[det_id]:
                # get from cache
                rmssscd1.append(rmssscd_cache[det_id][gt_id])
            else:  # compute from scratch
                # TODO: The nearest neighbors are also computed when making
                # the sim_cache.  It should be possible to only do
                # this computation once, but it will require a
                # fair amount of refactoring

                sub_points = evaluator._submission.elements[det_id].posed_points
                gt_points = evaluator._ground_truth.elements[gt_id].posed_points
                idx1to2, idx2to1, dist1to2, dist2to1 = nearest_neighbors(
                    sub_points[:, 0:3], gt_points[:, 0:3])

                color_diff1to2 = sub_points[:, 3:6] - gt_points[idx1to2, 3:6]
                color_diff2to1 = gt_points[:, 3:6] - sub_points[idx2to1, 3:6]

                n_matched = color_diff1to2.shape[0] + color_diff2to1.shape[0]

                if n_matched > 0:
                    # SUMO white paper Eq 13
                    rmssscd = np.sqrt(
                        (np.sum(color_diff1to2 * color_diff1to2) +
                         np.sum(color_diff2to1 * color_diff2to1)) / n_matched)
                else:
                    rmssscd = 0
                rmssscd_cache[det_id][gt_id] = rmssscd

                rmssscd1.append(rmssscd)

    if len(rmssscd1) > 0:
        return np.mean(rmssscd1)
    else:
        return math.inf  # no corrs found

#---------------------
# Visualization
#---------------------


def to_surface(mesh):
    """
    Convert Mesh object into pyny.Surface object (used for visualization)
    Inputs:
    mesh (Mesh)

    Return:
    pyny.Surface - converted mesh
    """
    vert = mesh.vertices
    faces = mesh.faces
    surface = []
    for i in range(faces.shape[0]):
        points = vert[faces[i], :]
        surface.append(pyny.Polygon(np.array(points)))
    return pyny.Surface(surface)


def visualize_mesh(mesh, visualize=True):
    """
    Visualize mesh.

    Inputs:
    mesh (Mesh) - mesh to visualize
    visualize (Boolean) - if False, mesh will not actually be
      visualized (this is useful for testing)

    """
    vertices = mesh.vertices().T.reshape((-1, 3))
    indices = mesh.indices().reshape((-1, 3))
    mesh = pymesh.form_mesh(vertices, indices)
    surface = to_surface(mesh)

    if visualize:
        surface.plot('r')
        plt.show()
        plt.waitforbuttonpress()
        plt.close()
