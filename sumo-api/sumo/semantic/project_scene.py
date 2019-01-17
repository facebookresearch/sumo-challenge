#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


Data class: Project scene file representation.
"""

import os
import xml.etree.ElementTree as ET
from sumo.semantic.project_object_dict import ProjectObjectDict

DEFAULT_SETTINGS = {
    "version": "2.0",
    "categories_id": "fb_categories_v1_0",
    "categories_url": "https://sumochallenge.org/en/categories-1_0.json",
}


class ProjectScene(object):
    """
    A representation for a project scene file hierarchy.
    Read and write project to disk.
    """

    def __init__(self, project_type, elements=None, settings=None):
        """
        Constructor

        Inputs:
        project_type (string) - Specifies the project type to construct (valid
           values are "bounding_box", "voxels", or "meshes")
        settings (dict) - stores version, categories, etc.
        elements (ProjectObjectDict)

        Exceptions:
        IOError - if scene cannot be read
        ValueError - if xml cannot be parsed
        ValueError - if project_type is not one of the allowed options
        """
        if project_type not in ["bounding_box", "voxels", "meshes"]:
            raise ValueError("Project type ({}) must be one of 'bounding_box', \
'voxels', or 'mesh'".format(project_type))
        self.project_type = project_type
        self.settings = settings if settings is not None else DEFAULT_SETTINGS
        self.elements = elements if elements is not None else ProjectObjectDict()

    def transform_pose(self, nTo):
        """
        Transform the pose of objects in the scene to a new pose.
        Return a new ProjectScene instance with the transformed pose.

        Input:  nTo (Pose3) -  pose transforming old to new coordinates
        Output: new_project_scene - new ProjectScene instance with
            transformed pose.
        """
        new_elements = {k: PO.transform_pose(nTo) for (k, PO) in self.elements.items()}
        return ProjectScene(project_type=self.project_type, elements=new_elements)

    def save(self, path, project_name):
        """
        Save project to disk.  Location will be <path>/<project_name>.

        Inputs:
        path (string) - path to base dir
        project_name (string) - project name

        Side effects:
        Creates a new directory <project_name>.

        Structure:
        <project_name>
            <project_name>.xml
            optional: [elements as glb files or voxel grids (h5 files)]

        Exceptions:
        RuntimeError - if <path>/<project_name> already exists.
        """

        # create <project_name> dir
        # Will raise OSError if <path>/<project_name> already exists
        project_dir = os.path.join(path, project_name)
        os.makedirs(project_dir)

        # create xml tree
        scene = ET.Element(
            "scene", attrib={"xmlns:sumo": "https://www.sumochallenge.org"}
        )

        # version section
        version = ET.SubElement(scene, "version")
        version.text = self.settings["version"]

        # categories section
        categories = ET.SubElement(scene, "categories")
        categories_id = ET.SubElement(categories, "id")
        categories_id.text = self.settings["categories_id"]
        categories_url = ET.SubElement(categories, "url")
        categories_url.text = self.settings["categories_url"]

        # project type
        project_type = ET.SubElement(scene, "project_type")
        project_type.text = self.project_type

        #  elements
        scene.append(self.elements.save(project_dir))

        # Write xml to disk
        xml_tree = ET.ElementTree(scene)
        filename = os.path.join(project_dir, project_name + ".xml")
        with open(filename, "wb") as file:
            xml_tree.write(file, encoding="utf-8", xml_declaration=True)

    @classmethod
    def load(cls, path, project_name):
        """
        Read project from disk location <path>/<project_name>

        Returns:
        Newly created ProjectScene instance.

        Exceptions:
        IOError - if project cannot be read
        ValueError - if xml cannot be parsed
        """
        settings = {}
        project_type = None

        project_dir = os.path.join(path, project_name)

        # Read xml from file
        xml_tree = ET.parse(os.path.join(project_dir, project_name + ".xml"))

        # parse xml tree
        root = xml_tree.getroot()
        if root.tag != "scene":
            raise ValueError('Expected root tag to be "scene"')

        # extract header info
        for element in root:
            if element.tag == "categories":
                for child_element in element:
                    if child_element.tag == "id":
                        settings["categories_id"] = child_element.text
                    elif child_element.tag == "url":
                        settings["categories_url"] = child_element.text
            elif element.tag == "project_type":
                project_type = element.text
            elif element.tag == "version":
                settings["version"] = element.text

            # Skip elements tag for now.
            # Any other tags are silently ignored

        if project_type is None:
            raise ValueError("xml is missing project_type tag")

        # extract elements
        for element in root:
            if element.tag == "elements":
                elements = ProjectObjectDict.load(
                    project_type=project_type, base_elem=element, path=project_dir)
            # header and any other tags are silently ignored

        return cls(project_type, elements, settings)
