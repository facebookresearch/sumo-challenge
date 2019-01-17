"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Support for mapping between categories and RGB colors.  This is useful
mainly for display purposes.
"""

import csv
import numpy as np
cimport numpy as np
import os


class ColoredCategory(object):
    """ Class to generate distinct RGB values for different categories of a dataset."""

    def __init__(self, csv_file):
        """ Create an object that holds color information of categories in
        <dataset_name> dataset.

        Inputs:
        csv_file (str) - filepath to an external csv file containing color mapping
            information.

        Exceptions:
        IOError - raised if csv_file does not exist.

        Internals:
        _data (list) -- contains a list of dictionaries. The index of the list
            corresponds to category index expected from a category image. The
            value of the list at each index contains a dictionary of all other
            info in the CSV file about that index, including RGB value.

        _reverse_data (list) -- contains pre-calculated inverse of the category
            look up table, where given a category name, one can find the category
            RGB value. This is useful when only dealing with category names and
            not an actual image with category index values.

        _lut (list) -- a 3xN numpy array where each column Y has a 3x1 RGB value
            for a category with index value of Y. This variable is exposed publicly
            through LUT property. It is populated with the first call to LUT
            property and cached thereafter. Do not use the private member _lut
            directly.

        Note:
        The CSV file must be in the the following format:
        id,category,is_evaluated,color

        example:
        id,category,is_evaluated,color
        0,empty,False,(152, 245, 255)
        1,accordion,False,(30, 144, 255)
        2,air_conditioner,True,(122, 197, 205)
        3,amplifier,False,(110, 123, 139)
        ...

        Note: the class will cast each R, G, B value into int. While there
        may not be an enforcement of range, the program expects to receive
        values between 0 - 255 for each element of R, G, and B.

        Note: The CSV file must have information about index values' RGB
        correspondence in exact order of the index sequence values. In other
        words, the class ignores the first column of the data, and assumes that
        the location of the RGB value corresponds to the index determined by row
        number. Ignoring the very first line of the CSV, the next line information
        corresponds to category index value of 0, regardless of what the first
        column shows. Because of this behavior, all category indices must have
        complete information in the CSV file.
        """
        if not os.path.exists(csv_file):
            raise IOError("File {} does not exist.".format(csv_file))

        with open(csv_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            self._data = [{
                "category": line["category"],
                "is_evaluated": line["is_evaluated"],
                "color": tuple(map(int, line["color"].split(";")))} for line in reader]

        self._reverse_data = {
            self._data[id]["category"]: self._data[id]["color"] for id, x in enumerate(self._data)
        }
        cdef np.ndarray lut = np.zeros((0))
        self._lut = lut

    @property
    def LUT(self):
        """ The lookup table for conversion of indexes to RGB values.

        Note: This method populate the internal _lut member upon first request, and
        returns that value. For subsequent calls, the cached value is returned.
        """
        if self._lut.shape[0] == 0:
            total = len(self._data)
            self._lut = np.array([x["color"]
                for x in self._data], dtype=np.uint8)
        return self._lut

    def category_id_to_rgb(self, category_id):
        """ Convert a category id to its corresponding color.
        Inputs:
        category_id (int) - id of the category

        Returns
        An RGB tuple, which corresponds to the category_id. The value of the tuple
        depends on the value provided in the CSV file, however it is expected that
        the values are between 0 and 255.

        Exceptions:
        KeyError - if category_id does not exist in the dataset config.
        """
        if category_id > len(self._data) or category_id < 0:
            raise KeyError("Category id {} does not exist.".format(category_id))
        return self._data[category_id]["color"]

    def category_name_to_rgb(self, category_name):
        """ Convert a category name to its corresponding color.
        Inputs:
        category_name (str) - name of the category

        Returns
        An RGB tuple, which corresponds to the category_name

        Exceptions:
        KeyError - if category_name does not exist in the dataset config.
        """
        if category_name not in self._reverse_data:
            raise KeyError("Category {} does not exist.".format(category_name))
        return self._reverse_data[category_name]

    def convert_to_rgb_im(self, np.ndarray indexed_im):
        """Converts an indexed_im to RGB based on category mapping.
        Inputs:
        indexed_im (NxM np.array of ints) - input indexed image. Each index value
            corresponds to category_id.

        returns
        An RGB image (NxMx3 np.array) where each index is converted to the
        corresponding RGB value

        Exceptions:
        IndexError -- if an index value in input image exists where the value is
        greater than the maximum index in the LUT.
        """
        return self.LUT[indexed_im]
