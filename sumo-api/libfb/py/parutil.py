#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os


def get_file_path(path):
    return os.path.join(os.getcwd(), path)


def get_dir_path(path):
    return os.path.join(os.getcwd(), path)
