import sys
import os
import csv
sys.path.append(os.getcwd())

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _init_paths
import functools
import torchnet as tnt
import torch

from data_utils import load_tiff
import numpy as np
from parse_args import parse_args

import scipy.io as sio
import pickle
import json
import uuid
import cv2

from sumo.semantic.project_scene import ProjectScene
from sumo.threedee.point_cloud import PointCloud
from sumo.threedee.textured_mesh import TexturedMesh
file_dir = os.path.dirname(os.path.realpath(__file__))


def _sample_element(element, density=625):
    counts = []
    for m in element.meshes.primitive_meshes():
        counts.append(len(m.indices()))
    n_faces3 = np.sum(counts)
    faces = np.zeros((n_faces3, 6))
    start = 0
    for i, mesh in enumerate(element.meshes.primitive_meshes()):
        temp = mesh.vertices()[:, mesh.indices()]
        faces[start:start+counts[i], 0:3] = temp.T
        if mesh.is_textured() and not (mesh.base_color().shape[0:2] == (0, 0)):
            base_color = mesh.base_color()
            uv_coords = mesh.uv_coords()[:, mesh.indices()]
            W, H, _ = base_color.shape
            ucoord = np.mod((uv_coords[0, :] * W).astype(int), W)
            vcoord = np.mod((uv_coords[1, :] * H).astype(int), H)
            faces[start:start+counts[i], 3:6] = base_color[ucoord, vcoord, :]
        start += counts[i]
    return _sample_mesh(faces, density)

def _sample_mesh(faces, density=625):
    A, B, C = faces[0::3, :], faces[1::3, :], faces[2::3, :]
    cross = np.cross(A[:, 0:3] - C[:, 0:3] , B[:, 0:3] - C[:, 0:3])
    areas = 0.5*(np.sqrt(np.sum(cross**2, axis=1)))
    Nsamples_per_face = (density*areas).astype(int)
    N = np.sum(Nsamples_per_face)
    if N == 0:
        return np.empty((0, 3))
    face_ids = np.zeros((N,), dtype=int)
    count = 0
    for i, n in enumerate(Nsamples_per_face):
        face_ids[count:count + Nsamples_per_face[i]] = i
        count += Nsamples_per_face[i]
    A = A[face_ids, :]; B = B[face_ids, :]; C = C[face_ids, :]
    r = np.random.uniform(0, 1, (N, 2))
    sqrt_r1 = np.sqrt(r[:, 0:1])
    samples = (1 - sqrt_r1)*A + sqrt_r1*(1 - r[:, 1:])*B + sqrt_r1*r[:, 1:]*C
    return samples

def extract_bbox(mask):
    horizontal_indices = np.where(np.any(mask, axis=0))[0]
    vertical_indices = np.where(np.any(mask, axis=1))[0]
    if horizontal_indices.shape[0]:
        x1, x2 = horizontal_indices[[0, -1]]
        y1, y2 = vertical_indices[[0, -1]]
        if x1 == x2 and y1 == y2:
            x1, x2, y1, y2 = 0, 0, 0, 0
    else:
        x1, x2, y1, y2 = 0, 0, 0, 0
    return x1, y1, x2, y2


def get_single_view_data(slice_ind, data, H, W, scene, fname, args):
    slice_data = data[:, slice_ind*H:(slice_ind+1)*H, :W] # Only use one frame

    im_data = slice_data[0:3, :, :]

    instances = np.unique(slice_data[5, :, :])
    num_objs = 0

    gt_boxes = []
    masks = []
    class_ids = []
    Rs = []
    Ts = []

    for instance in instances:
        mask_data = slice_data[5, :, :] == instance
        loc = np.where(slice_data[5, :, :] == instance)
        y, x = loc[0][0], loc[1][0]
        class_value = slice_data[4, y, x]
        if class_value == 0: # empty/background
            continue

        obj = scene.elements[str(int(instance))]

        num_objs += 1
        x1, y1, x2, y2 = extract_bbox(mask_data)
        class_ids.append(class_value)
        masks.append(mask_data)
        gt_boxes.append([x1, y1, x2, y2, class_value])
        Rs.append(obj.pose.R.R.flatten())
        Ts.append(obj.pose.t)

    if len(gt_boxes) > 0:
        masks = np.stack(masks, axis=0).astype(np.bool)
        class_ids = np.array(class_ids, dtype=np.int32)
        gt_boxes = np.array(gt_boxes, dtype=np.int32)
        Rs = np.stack(Rs)
        Ts = np.stack(Ts)

        #num_objs = len(gt_boxes)

        # Resize to args.scene_H, args.scene_W
        im_data = cv2.resize(im_data.T, (args.scene_W, args.scene_H)).T
        gt_boxes[:, :2] = np.floor(gt_boxes[:, :2] * (args.scene_H * 1. / H))
        gt_boxes[:, 2:4] = np.ceil(gt_boxes[:, 2:4] * (args.scene_H * 1. / H))

        metadata = ['{}.{:d}'.format('/'.join(fname.split("/")[-2:]), 0)]
        return metadata, im_data[np.newaxis,...], gt_boxes, masks, Rs, Ts
    else:
        im_data = cv2.resize(im_data.T, (args.scene_W, args.scene_H)).T
        metadata = ['{}.{:d}'.format('/'.join(fname.split("/")[-2:]), 0)]
        return [metadata, im_data[np.newaxis,...]] + [None,]*4

def sumo_config(args):
    args.DATA_PATH = '%s/../data/%s' % (file_dir, args.dataset)
    args.CONFIG_PATH = '%s/../config/' % (file_dir)
    args.categories_map = get_sumo_mapping(args)
    args.categories = get_categories(args)


def loader(fname, train, args, db_path):

    # Load point clouds
    folder = fname[fname.rfind("/")+1:fname.rfind(".")]
    scene = ProjectScene.load(db_path + "/training_ground_truth", folder)

    data = load_tiff(fname, args).T

    W = data.shape[2]
    H = data.shape[1] // 6
    assert(W == H)
    assert(args.scene_H == args.scene_W)
    Nout = 6 # Number of outputs in Data Loader
    results = [[] for i in range(Nout)]

    # Use first one with non-zero number of instances
    for slice_ind in range(6):
        res = get_single_view_data(slice_ind, data, H, W, scene, fname, args)
        # metadata
        results[0] += res[0]
        for i in range(1, len(res)):
            if res[i] is not None:
                results[i].append(res[i])
    for i in range(1, len(res)):
        results[i] = np.concatenate(results[i], 0)[np.newaxis, ...]

    return results


def collate(batch):
    """ Collates a list of dataset samples into a single batch
    """
    metadata, im_data, gt_boxes, masks, Rs, Ts = list(zip(*batch))
    if len(metadata[0])>0:
        im_data = np.concatenate(im_data, 0)
        b, n, c, s1, s2 = im_data.shape
        im_data = im_data.swapaxes(2, 4).reshape(b, n*s2, s1, c).swapaxes(1, 3)
        gt_boxes = np.concatenate(gt_boxes, 0)
        masks = np.concatenate(masks, 0)
        Rs = np.concatenate(Rs, 0)
        Ts = np.concatenate(Ts, 0)
        metadata = [item for sublist in metadata for item in sublist]
    return metadata, im_data, gt_boxes, masks, Rs, Ts

def get_datasets(args):
    """ Gets training and test datasets. """
    trainlist, vallist, testlist = [], [], []
    trainlist = [os.path.join(args.DATA_PATH, 'training_input/', k.rstrip()+ '.tiff') for k in open(args.DATA_PATH + '/train.txt').readlines()][0:2]
    vallist = [os.path.join(args.DATA_PATH, 'training_input/', k.rstrip()+ '.tiff') for k in open(args.DATA_PATH + '/train.txt').readlines()][0:2]
    testlist = [os.path.join(args.DATA_PATH, 'training_input/', k.rstrip()+ '.tiff') for k in open(args.DATA_PATH + '/train.txt').readlines()][0:2]
    return tnt.dataset.ListDataset(sorted(trainlist), functools.partial(
        loader, train=True, args=args, db_path=args.DATA_PATH)),\
        tnt.dataset.ListDataset(sorted(vallist), functools.partial(
            loader, train=False, args=args, db_path=args.DATA_PATH)),\
        tnt.dataset.ListDataset(sorted(testlist), functools.partial(
            loader, train=False, args=args, db_path=args.DATA_PATH))

def get_sumo_mapping(args):
    mapping = {}
    category_id = 0
    categories_path = os.path.join(args.CONFIG_PATH, 'sumo_categories.csv')
    with open(categories_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if row[2] == "True" or category_id == 0: # keep "Empty" as id = 0
                mapping[int(row[0])] = category_id
                category_id += 1
            else:
                mapping[int(row[0])] = 0 # set to Empty category if is excludable
    return mapping

def get_categories(args):
    class_names = []
    with open(os.path.join(args.CONFIG_PATH, 'sumo_classes.txt'), 'r') as f:
        class_names = json.load(f)
    return class_names

# Given inputs:
#   single image
#   multiple bounding boxes
#   multiple points
# Returns:
#   multiple image
#   multiple points

def test_loader():

    args = type('obj', (object,), {})
    args.show = False
    args.dataset = 'SUMO'
    args.H = 1024 # Load at original resolution
    args.W = 1024

    args.scene_H = 1024
    args.scene_W = 1024
    args.num_points = 2048

    args.model_W = 256
    args.model_H = 192

    args.batch_size = 1
    args.CONFIG_PATH = 'config/'
    args.DATA_PATH = 'data/%s' % (args.dataset)
    args.collate = collate
    args.categories_map = get_sumo_mapping(args)
    args.categories = get_categories(args)

    train_dataset, val_dataset, test_dataset = get_datasets(args)
    loader_ = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                        collate_fn=collate, num_workers=0,
                                        shuffle=False, drop_last=True)

    # Batch index
    for bidx, (metadata, im_data, gt_boxes, masks, Rs, Ts) in enumerate(loader_):

        # Image/scene index in batch
        for iidx in range(args.batch_size):

            im = im_data[iidx]
            boxes = gt_boxes[iidx]
            print(im.shape, Rs[iidx].shape, Ts[iidx].shape, masks[iidx].shape)
            """
            plt.imshow(im.swapaxes(0,2).astype(int))
            plt.waitforbuttonpress()
            plt.close()
            """

if __name__ == '__main__':
    test_loader()

