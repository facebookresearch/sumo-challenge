#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

ObjectSymmetry unit tests.
"""

import unittest
import xml.etree.cElementTree as ET

from sumo.semantic.object_symmetry import ObjectSymmetry, SymmetryType


class TestObjectSymmetry(unittest.TestCase):
    """Unit tests for ObjectSymmetry class"""

    def test_creation(self):
        """Test basic functionality: default creation, general creation, and
        attribute access."""
        sym = ObjectSymmetry()
        self.assertEqual(sym.x_symmetry, SymmetryType.none)
        self.assertEqual(sym.y_symmetry, SymmetryType.none)
        self.assertEqual(sym.z_symmetry, SymmetryType.none)

        sym = ObjectSymmetry(
            SymmetryType.twoFold, SymmetryType.twoFold, SymmetryType.fourFold
        )
        self.assertEqual(sym.x_symmetry, SymmetryType.twoFold)
        self.assertEqual(sym.y_symmetry, SymmetryType.twoFold)
        self.assertEqual(sym.z_symmetry, SymmetryType.fourFold)

    def test_invalid_creation(self):
        self.assertRaises(
            ValueError,
            ObjectSymmetry,
            x_symmetry=SymmetryType.spherical,
            y_symmetry=SymmetryType.spherical,
            z_symmetry=SymmetryType.none,
        )

    def test_is_valid(self):
        """Test the is_valid function"""
        self.assertTrue(
            ObjectSymmetry.is_valid(
                SymmetryType.cylindrical, SymmetryType.twoFold, SymmetryType.twoFold
            )
        )
        self.assertFalse(
            ObjectSymmetry.is_valid(
                SymmetryType.none, SymmetryType.cylindrical, SymmetryType.fourFold
            )
        )

    def test_from_strings(self):
        """Test creation from strings"""
        strings = ["none", "none", "twoFold"]
        sym = ObjectSymmetry.from_strings(strings)
        expected_sym = ObjectSymmetry(
            SymmetryType.none, SymmetryType.none, SymmetryType.twoFold)
        self.assertEqual(sym, expected_sym)

    def test_xml(self):
        """Test converting to and from xml."""

        s = b"<symmetry><x>twoFold</x><y>cylindrical</y><z>twoFold</z></symmetry>"
        expected_sym = ObjectSymmetry(
            SymmetryType.twoFold, SymmetryType.cylindrical, SymmetryType.twoFold
        )

        # xml -> object
        xml = ET.fromstring(s)
        sym = ObjectSymmetry.from_xml(xml)
        print("sym=", sym)
        print("expected=", expected_sym)
        self.assertEqual(sym, expected_sym)

        # object -> xml
        xml = sym.to_xml()
        self.assertEqual(ET.tostring(xml), s)
