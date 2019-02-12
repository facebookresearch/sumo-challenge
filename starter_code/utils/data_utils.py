import imageio
import numpy as np
import random
import math
import sys, os
import pandas as pd
import cv2
file_dir = os.path.dirname(os.path.realpath(__file__))

def load_csv(path, verbose=False):
    if verbose:
        print("Loading %s \n" % (path))
    return pd.read_csv(path, delim_whitespace=True, header=None).values

def load_tiff(fname, args):
    im = imageio.mimread(fname)
    labels = ["RGB", "Depth", "Category", "Instance"]
    data = []
    for i in range(len(labels)):
        img = im[i]
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        data.append(cv2.resize(img.astype('float'), (6*args.H, args.W)))
        if len(data[-1].shape) == 2:
            data[-1] = data[-1][:, :, np.newaxis]
        if args.categories_map and labels[i] == "Category":
            data[-1] = np.vectorize(args.categories_map.get)(data[-1].astype(int))
    data = np.concatenate(data, axis=2)
    return data
