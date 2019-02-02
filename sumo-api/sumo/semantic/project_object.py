#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


Data class: Object in a project.
"""

from copy import deepcopy
import numpy as np
import os
import xml.etree.cElementTree as ET

from sumo.base.vector import Vector3
from sumo.geometry.pose3 import Pose3
from sumo.semantic.object_symmetry import ObjectSymmetry
from sumo.threedee.box_3d import Box3d
from sumo.threedee.compute_bbox import ComputeBbox
from sumo.threedee.gltf_model import GltfModel
from sumo.threedee.voxel_grid import VoxelGrid


class ProjectObject(object):
    """
    Object in a project.  Depending on SUMO track, one of the following sets is
    stored:
    bounding box track: bounds
    voxels track: voxels
    mesh track: mesh

    Public attributes:
        id (string) (read only) - Unique identifier for the object
        project_type (string) (read only) - Specifies the project type to construct
            (valid values are "bounding_box", "voxels", or "meshes")
        bounds (Box3d) (read only) - Axis-aligned bounding box of the object.
        voxels (VoxelGrid) - Represents object voxelized shape
        meshes (GltfModel) - Represents object mesh shape and appearance in local
           coordinates
        pose (Pose3) - Rigid body transform that transforms object from local
          coordinates to world coordinates
        category (string) - Object category (e.g., chair, bookcase, etc.)
        symmetry (ObjectSymmetry) - Object symmetry description
        score (float) - If object is generated from a detector,
          this is the score.  Higher is better.  Value is -1 if not used.
        evaluated (Boolean) - If object is part of ground truth, this indicates
          whether the object is going to be used as part of the evaluation
          metric for the scene it is part of.  Otherwise, this attribute's meaning
          is undefined.
    """

    def __init__(
        self, id, project_type="bounding_box", bounds=None, voxels=None, meshes=None,
        pose=None, category="unknown", symmetry=None, score=-1, evaluated=True
    ):
        """
        Constructor.  Preferred method of creation is one of the factory methods:
        gen_bounding_box_object, gen_voxels_object, or gen_meshes_object.

        For this constructor, representation is selected based on the <project_type>:
        bounding box track: bounds is used
        voxels track: voxels and bounds are used
        mesh track: meshes and bounds are used

        Inputs:
        id (string) - Unique identifier for the object
        project_type (string) - Specifies the project type to construct (valid
            values are "bounding_box", "voxels", or "meshes")
        bounds (Box3d) - Object bounding box in local coordinates
        voxels (VoxelGrid) - Object voxel shape in local coordinates
        meshes (GltfModel) - Object mesh shape and appearance in local coordinates
        pose (Pose3) - Transforms object from local coordinates to world coordinates
        category (string) - Object category (e.g., chair, bookcase, etc.)
        symmetry (ObjectSymmetry) - Object symmetry description
        score (float) - Detection score
        evaluated (Boolean) - Indicates whether this object will be used in
            evaluation metric.  Only relevant for ground truth scenes.

        Exceptions:
            ValueError - if project_type is not one of the allowed values.
        """
        # ensure id is unicode string, idiom below is python2/3 compatible
        self._id = id.decode('UTF-8') if hasattr(id, 'decode') else id
        self._project_type = project_type
        self.pose = pose if pose is not None else Pose3()
        self.category = category
        self.symmetry = symmetry if symmetry is not None else ObjectSymmetry()
        self.score = score
        self.evaluated = evaluated

        if project_type == "bounding_box":
            self.bounds = bounds
            self.voxels = None
            self.meshes = None
        elif project_type == "voxels":
            self.bounds = bounds
            self.voxels = voxels
            self.meshes = None
        elif project_type == "meshes":
            self.bounds = bounds
            self.voxels = None
            self.meshes = meshes
        else:
            raise ValueError("Invalid project_type: " + project_type)

    @classmethod
    def gen_bounding_box_object(cls, id, bounds=None, pose=None, category="unknown",
    symmetry=None, score=-1, evaluated=True):
        """
        Factory method for making a ProjectObject that holds a bounding_box.

        Inputs:
        See constructor.

        Return:
        new ProjectObject (project_type = "bounding_box")
        """
        return cls(id, "bounding_box", bounds=bounds, voxels=None, meshes=None,
            pose=pose, category=category, symmetry=symmetry, score=score,
            evaluated=evaluated)

    @classmethod
    def gen_voxels_object(cls, id, bounds=None, voxels=None,
          pose=None, category="unknown", symmetry=None, score=-1, evaluated=True):
        """
        Factory method for making a ProjectObject that holds a voxel grid.

        Inputs:
        See constructor.

        Return:
        new ProjectObject (project_type = "voxels")
        """
        return cls(id, "voxels", bounds=bounds, voxels=voxels, meshes=None,
            pose=pose, category=category, symmetry=symmetry, score=score,
            evaluated=evaluated)

    @classmethod
    def gen_meshes_object(cls, id, bounds=None, meshes=None,
          pose=None, category="unknown", symmetry=None, score=-1,
          evaluated=True):
        """
        Factory method for making a ProjectObject that holds a mesh model.

        Inputs:
        See constructor.

        Return:
        new ProjectObject (project_type = "meshes")
        """
        return cls(id, "meshes", bounds=bounds, voxels=None, meshes=meshes,
            pose=pose, category=category, symmetry=symmetry, score=score,
            evaluated=evaluated)

    # Restrict id to be read only.
    @property
    def id(self):
        return self._id

    # Restrict project_type to be read only.
    @property
    def project_type(self):
        return self._project_type

    def almost_equal(self, other):
        """ Check that two instances are almost equal.
            Note: just checks object's size.
        """
        if (self.meshes is not None) and (not self.meshes.same_size(other.meshes)):
            return False

        return (
            self.id == other.id
            and self.pose.almost_equal(other.pose)
            and self.category == other.category
            and self.symmetry == other.symmetry
            and np.isclose(self.score, other.score)
            and self.evaluated == other.evaluated
        )

    def transform_pose(self, nTo):
        """
        Transform the pose of object in the scene to a new pose.

        Input:  nTo (Pose3) -  pose transforming old to new coordinates
        Output: new ProjectObject instance with the transformed pose.
        """
        return ProjectObject(self.id, self.project_type,
            self.bounds, self.voxels, self.meshes,
            nTo.compose(self.pose), self.category, self.symmetry, self.score,
            self.evaluated
        )

    def save(self, path=None):
        """
        Save self to disk.  Also creates an xml Element with the appropriate tags.
        If <project_type> is bounding_box, no files are written - only xml Element
        is generated, so <path> is not used.

        Inputs:
        path (string) - path to project directory (if project_type is voxels or
           mesh).

        Return:
        Element whose tag is "object" with appropriate sub-elements
        (id, category, etc.)

        Exceptions:
        IOError - if file cannot be written.
        """

        base_elem = self._to_xml()

        # write out files
        if self.project_type == "meshes":
            filename = os.path.join(path, self.id + ".glb")
            self.meshes.save_as_glb(filename)
        elif self.project_type == "voxels":
            filename = os.path.join(path, self.id + ".h5")
            self.voxels.save(filename)

        return base_elem

    @classmethod
    def load(cls, project_type, base_elem, path=None):
        """
        Factory method to create a ProjectObject by reading it from disk and
        xml.
        Note: If <project_type> is bounding_box, no file is read and <path>
        is not relevant.

        Inputs:
        project_type (string) - Specifies the project type to construct (valid
            values are "bounding_box", "voxels", or "meshes")
        base_elem (ET.Element) - An Element with tag "element" and appropriate
            sub-elements.
        path (string) - path to project directory (only required if project_type
            is voxels or mesh).

        Return:
        New ProjectObject

        Exceptions:
        ValueError - If base_elem is not <element> or if none of its children is <id>.
        ValueError - If project_type is not valid.
        IOError - if file cannot be read.
        """
        (id, pose, category, bounds, symmetry, score, evaluated) = \
            cls._parse_xml(base_elem)

        # load file-based attributes and return the constructed object
        if project_type == "bounding_box":
            return ProjectObject.gen_bounding_box_object(
                id=id, bounds=bounds, pose=pose, category=category,
                symmetry=symmetry, score=score, evaluated=evaluated)
        elif project_type == "voxels":
            voxels = VoxelGrid.from_file(os.path.join(path, id + ".h5"))
            return ProjectObject.gen_voxels_object(
                id=id, bounds=bounds, voxels=voxels, pose=pose, category=category,
                symmetry=symmetry, score=score, evaluated=evaluated)
        elif project_type == "meshes":
            meshes = GltfModel.load_from_glb(os.path.join(path, id + ".glb"))
            return ProjectObject.gen_meshes_object(
                id=id, bounds=bounds, meshes=meshes, pose=pose, category=category,
                symmetry=symmetry, score=score, evaluated=evaluated)
        else:
            raise ValueError("Invalid project_type: " + project_type)

    @classmethod
    def example(cls, id="1"):
        """Create a simple ProjectObject of project_type = meshes."""
        meshes = GltfModel.example()
        pose = Pose3(t=Vector3(1, 2, 3))
        symmetry = ObjectSymmetry.example()
        return cls.gen_meshes_object(id=id, pose=pose, category="chair", meshes=meshes,
            symmetry=symmetry, score=0.57, evaluated=False)

#-------------
# End of public interface

    def _to_xml(self):
        """
        Convert object to xml.
        Return:
            Element whose tag is "object" with appropriate sub-elements
            (id, category, bounds, and pose)
        """
        base_elem = ET.Element("element")

        id_elem = ET.SubElement(base_elem, "id")
        id_elem.text = self.id

        category_elem = ET.SubElement(base_elem, "category")
        category_elem.text = self.category

        if self.project_type == "meshes":
            compute_bbox_alg = ComputeBbox()
            bounds = compute_bbox_alg.from_meshes(self.meshes.primitive_meshes())
        elif self.project_type == "voxels":
            bounds = self.voxels.bounds()
        elif self.project_type == "bounding_box":
            bounds = self.bounds
        bounds_xml = bounds.to_xml()
        bounds_xml.tag = "bounds"
        base_elem.append(bounds_xml)

        base_elem.append(self.pose.to_xml())

        base_elem.append(self.symmetry.to_xml())

        score_elem = ET.SubElement(base_elem, "detectionScore")
        score_elem.text = str(self.score)

        evaluated_elem = ET.SubElement(base_elem, "evaluated")
        evaluated_elem.text = str(self.evaluated)

        return base_elem

    @staticmethod
    def _parse_xml(base_elem):
        """
        Parse the xml of an <element> tag, extracting the ProjectObject attributes.

        Inputs:
        base_elem (ET.Element) - An Element with tag "element" and appropriate
            sub-elements.

        Return:
        tuple (id, pose, category, bounds, symmetry, score, evaluated)
        ProjectObject attributes (see constructor for details).

        Exceptions:
        ValueError - If base_elem is not <element> or if none of its children is <id>.
        """
        # verify base_elem tag is 'element'
        if base_elem.tag != "element":
            raise ValueError('Expected tag to be "element"')

        # defaults
        proxy = ProjectObject(1)
        category = proxy.category
        pose = proxy.pose
        symmetry = proxy.symmetry
        score = proxy.score
        evaluated = proxy.evaluated

        for elem in base_elem:
            if elem.tag == "id":
                id = elem.text
            elif elem.tag == "pose":
                pose = Pose3.from_xml(elem)
            elif elem.tag == "category":
                category = elem.text
            elif (elem.tag == "bounds"):
                # Note: Boxe3d.from_xml expects tag to be 'box3d'
                elem_temp = deepcopy(elem)
                elem_temp.tag = "box3d"
                bounds = Box3d.from_xml(elem_temp)
            elif elem.tag == "symmetry":
                symmetry = ObjectSymmetry.from_xml(elem)
            elif elem.tag == "detectionScore":
                score = float(elem.text)
            elif elem.tag == "evaluated":
                evaluated = elem.text in ["True", "true"]

        if id is None:
            raise ValueError("XML is missing required <id> tag.")

        return (id, pose, category, bounds, symmetry, score, evaluated)
