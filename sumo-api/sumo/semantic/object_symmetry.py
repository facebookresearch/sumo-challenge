#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Support for symmetry of objects.  We support a limited subset of all possible
symmetries. Specifically, we only consider rotational symmetries about the
coordinate axes, and only certain types of symmetries are allowed (2-fold
rotation, 4-fold, and infinite (cylindrical or spherical)
"""

import xml.etree.cElementTree as ET
from enum import Enum


class SymmetryType(Enum):
    """
    Possible symmetries for a given axis.
    """

    none = 0
    twoFold = 1  # e.g., rectangle
    fourFold = 2  # e.g., square
    cylindrical = 3  # e.g., can
    spherical = 4  # e.g., ball


class ObjectSymmetry(object):
    """
    Symmetry representation for objects.
    Stores symmetry type along x, y, and z axes.

    Public attributes:
       x_symmetry (SymmetryType) - symmetry along x axis (read only)
       y_symmetry (SymmetryType) - symmetry along y axis (read only)
       z_symmetry (SymmetryType) - symmetry along z axis (read only)
    """

    def __init__(
        self,
        x_symmetry=SymmetryType.none,
        y_symmetry=SymmetryType.none,
        z_symmetry=SymmetryType.none,
    ):
        """
        Constructor.

        Inputs:
            x_symmetry, y_symmetry, z_symmetry (all SymmetryType) - symmetry
            along respective axis

        Exceptions:
            ValueError - if the given combination of symmetry is not allowed.

        Only a limited subset of combinations of symmetry on the different axes
        is allowed.  Specifically,
        1) no symmetry
        2) 1 twofold symmetry.  e.g., rectangular table
        3) 1 fourfold symmetry.  e.g., square table
        4) 2 twofold and 1 fourfold.  e.g., box with 2 equal side legnths
        5) 3 fourfold.  e.g., cube
        6) 3 twofold.  e.g., box with all unequal side lengths
        7) 1 cylindrical.  e.g., tapered drinking glass
        8) 1 cylindrical and 2 twofold.  e.g., cylinder
        9) all 3 spherical

        All non-specified symmetry axes are implicitly 'none'.
        """
        if not ObjectSymmetry.is_valid(x_symmetry, y_symmetry, z_symmetry):
            raise ValueError("Invalid combination of symmetry types")

        self._x_symmetry = x_symmetry
        self._y_symmetry = y_symmetry
        self._z_symmetry = z_symmetry

    def __str__(self):
        """
        Print in human-readable format.
        """
        return "x_symmetry: {}, y_symmetry: {}, z_symmetry: {}".format(
            self._symmetry_type_to_text[self.x_symmetry],
            self._symmetry_type_to_text[self.y_symmetry],
            self._symmetry_type_to_text[self.z_symmetry],
        )

    def __eq__(self, other):
        return (
            self.x_symmetry == other.x_symmetry
            and self.y_symmetry == other.y_symmetry
            and self.z_symmetry == other.z_symmetry
        )

    def __ne__(self, other):
        return self != other

    @property
    def x_symmetry(self):
        return self._x_symmetry

    @property
    def y_symmetry(self):
        return self._y_symmetry

    @property
    def z_symmetry(self):
        return self._z_symmetry

    @staticmethod
    def is_valid(x_symmetry, y_symmetry, z_symmetry):
        """
        Return true iff the inputs are a valid combination of symmetries (see
        rules in constructor).

        Inputs:
            x_symmetry, y_symmetry, z_symmetry (all SymmetryType) - symmetry
            along respective axis

        Return:
            Boolean
        """
        return (x_symmetry, y_symmetry, z_symmetry) in _valid_symmetries

    @classmethod
    def from_strings(cls, strings):
        """
        Create ObjectSymmetry instance from list of strings.
        Input:
            strings (list of strings) - text description of symmetries
        Return:
            ObjectSymmetry class representing the symmetries
        Exceptions:
            ValueError if there are not 3 strings in the list, or if
            the strings do not represent a known symmetry
        """
        if len(strings) != 3:
            raise ValueError(
                "ObjectSymmetry: expected 3 strings, got {}".format(len(strings))
            )
        symmetries = [
            cls._symmetry_text_to_type[s]
            for s in strings
            if s in cls._symmetry_text_to_type
        ]
        if len(symmetries) < 3:
            raise ValueError("ObjectSymmetry: unknown symmetry {}".format(strings))
        return cls(symmetries[0], symmetries[1], symmetries[2])

    @classmethod
    def from_xml(cls, base_elem):
        """
        Create ObjectSymmetry instance from xml tree <base_elem>.
        Format of xml is:
        <symmetry>
          <x>symmetryTypeText</x>
          <y>symmetryTypeText</y>
          <z>symmetryTypeText</z>
        </symmetry>

        Input:
            base_elem - cElementtree element with <symmetry> tag.

        Return:
            ObjectSymmetry instance

        Exceptions:
            RuntimeError - if xml tree cannot be parsed or if xml tags violate
              the rules.
        """

        if base_elem.tag != "symmetry":
            raise RuntimeError(
                "Expected 'symmetry' tag but got {}".format(base_elem.tag)
            )

        # parse the child tags
        x_symmetry = y_symmetry = z_symmetry = None
        for elem in base_elem:
            if elem.tag == "x":
                x_symmetry = cls._symmetry_text_to_type.get(elem.text, None)
            elif elem.tag == "y":
                y_symmetry = cls._symmetry_text_to_type.get(elem.text, None)
            elif elem.tag == "z":
                z_symmetry = cls._symmetry_text_to_type.get(elem.text, None)

        # create and return objectSymmetry instance
        try:
            objectSymmetry = cls(x_symmetry, y_symmetry, z_symmetry)
            return objectSymmetry

        except ValueError:
            RuntimeError("Could not parse symmetry element.")

    def to_xml(self):
        """
        Convert ObjectSymmetry instance to xml (see above for format)

        Return:
            Element containing <symmetry> tag
        """

        base_elem = ET.Element("symmetry")
        x_elem = ET.SubElement(base_elem, "x")
        x_elem.text = self._symmetry_type_to_text[self.x_symmetry]
        y_elem = ET.SubElement(base_elem, "y")
        y_elem.text = self._symmetry_type_to_text[self.y_symmetry]
        z_elem = ET.SubElement(base_elem, "z")
        z_elem.text = self._symmetry_type_to_text[self.z_symmetry]

        return base_elem

    @classmethod
    def example(cls):
        """Create a simple ObjectSymmetry instance for testing"""
        return cls(SymmetryType.twoFold, SymmetryType.cylindrical, SymmetryType.twoFold)

    # ---------------------------------
    # End of public interface

    # look up tables from enum to string and back
    _symmetry_type_to_text = {
        SymmetryType.none: "none",
        SymmetryType.twoFold: "twoFold",
        SymmetryType.fourFold: "fourFold",
        SymmetryType.cylindrical: "cylindrical",
        SymmetryType.spherical: "spherical",
    }

    _symmetry_text_to_type = {
        "none": SymmetryType.none,
        "twoFold": SymmetryType.twoFold,
        "fourFold": SymmetryType.fourFold,
        "cylindrical": SymmetryType.cylindrical,
        "spherical": SymmetryType.spherical,
    }


# list of valid symmetries
_valid_symmetries = {
    # no symmetry
    (SymmetryType.none, SymmetryType.none, SymmetryType.none),
    # 1 twofold symmetry.  e.g., rectangular table
    (SymmetryType.twoFold, SymmetryType.none, SymmetryType.none),
    (SymmetryType.none, SymmetryType.twoFold, SymmetryType.none),
    (SymmetryType.none, SymmetryType.none, SymmetryType.twoFold),
    # 1 fourfold symmetry.  e.g., square table
    (SymmetryType.fourFold, SymmetryType.none, SymmetryType.none),
    (SymmetryType.none, SymmetryType.fourFold, SymmetryType.none),
    (SymmetryType.none, SymmetryType.none, SymmetryType.fourFold),
    # 2 twofold and 1 fourfold.  e.g., box with 2 equal side legnths
    (SymmetryType.fourFold, SymmetryType.twoFold, SymmetryType.twoFold),
    (SymmetryType.twoFold, SymmetryType.fourFold, SymmetryType.twoFold),
    (SymmetryType.twoFold, SymmetryType.twoFold, SymmetryType.fourFold),
    # 3 fourfold.  e.g., cube
    (SymmetryType.fourFold, SymmetryType.fourFold, SymmetryType.fourFold),
    # 3 twofold.  e.g., box with all unequal side lengths
    (SymmetryType.twoFold, SymmetryType.twoFold, SymmetryType.twoFold),
    # 1 cylindrical.  e.g., tapered drinking glass
    (SymmetryType.cylindrical, SymmetryType.none, SymmetryType.none),
    (SymmetryType.none, SymmetryType.cylindrical, SymmetryType.none),
    (SymmetryType.none, SymmetryType.none, SymmetryType.cylindrical),
    # 1 cylindrical and 2 twofold.  e.g., cylinder
    (SymmetryType.cylindrical, SymmetryType.twoFold, SymmetryType.twoFold),
    (SymmetryType.twoFold, SymmetryType.cylindrical, SymmetryType.twoFold),
    (SymmetryType.twoFold, SymmetryType.twoFold, SymmetryType.cylindrical),
    # all 3 spherical
    (SymmetryType.spherical, SymmetryType.spherical, SymmetryType.spherical),
}
