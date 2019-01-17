"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

GlTF to GLB converter.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import mimetypes
import os
import struct
import sys


class BodyEncoder:
    """
    Encode the binary chunks of the GLB's body
    Use body_length and body_parts for tracking aligned binary data
    """
    def __init__(self, containing_dir):
        """
        Constructor

        Input:
        containing_dir (string) - Path for gltf file
        """
        self.containing_dir = containing_dir
        self.body_length = 0
        self.body_parts = []

    def add_to_body(self, uri, length):
        """
        Adds an immediate or external data uri to the GLB's body

        Inputs:
        uri (string) - String of characters used to identify a resource
        length (int) - Placeholder to store buffer length

        Outputs:
        offset (int) - Overall offset to current uri
        length (int) - Length for current aligned uri buffer
        """
        with open(os.path.join(self.containing_dir, uri), 'rb') as f:
            buf = bytearray(f.read())

        # 4-byte-aligned
        length = (len(buf) + 3) & ~3
        for _i in range(length - len(buf)):
            buf.extend(b' ')

        # Handle the buffer
        offset = self.body_length
        self.body_parts.append(offset)
        self.body_parts.append(buf)
        length = len(buf)
        self.body_length += length
        return (offset, length)


class GlbEncoder:
    """
    Combine JSON header and binary body into a GLB with a proper header
    """
    def __init__(self, header, body):
        """
        Constructor

        Inputs:
        header - Packed gltf json in binary data
        body (BodyEncoder) - Packed gltf resource in binary data
        """
        self.header = header
        self.body = body

    def export(self, filename):
        """
        Export the GLB file

        Inputs:
        filename (string) - Path for exported file path
        """
        with open(filename, 'wb') as f:
            f.write(self.export_string())
            f.close()

    def export_string(self):
        """ Export the GLB data"""

        # Padded scene length to a multiple of 4 for 4-byte-aligned
        l_aligned_len = (len(self.header) + 3) & ~3

        for _i in range(l_aligned_len - len(self.header)):
            self.header.extend(b' ')

        glb_out = bytearray()
        # 12 (file header): magic + version + length
        # 8 (json chunk header): json length + type
        # len(self.header): aligned json header length
        # 8 (bin chunk header): chunk length + type
        # self.body.body_length: aligned binary length
        file_len = 12 + 8 + len(self.header) + 8 + self.body.body_length

        # Write the header
        glb_out.extend(struct.pack('<I', 0x46546C67))   # magic number: "glTF"
        glb_out.extend(struct.pack('<I', 2))
        glb_out.extend(struct.pack('<I', file_len))
        glb_out.extend(struct.pack('<I', len(self.header)))

        # Write binary json
        glb_out.extend(struct.pack('<I', 0x4E4F534A))
        glb_out += self.header
        glb_out.extend(struct.pack('<I', self.body.body_length))
        glb_out.extend(struct.pack('<I', 0x004E4942))

        # Write the body
        for i in range(0, len(self.body.body_parts), 2):
            contents = self.body.body_parts[i + 1]
            glb_out += contents

        return glb_out


def gltf2glb(input_glTF, output_glb=None):
    """
    Convert glTF2 files into a single glb file

    Inputs:
    input_glTF (string) - Path for input gltf file path
    output_glb (string) - Path for output glb file path
    """
    # Check if input file is with gltf extension
    if not input_glTF.endswith('.gltf'):
        print("Failed to create binary GLTF file: input is not *.gltf")
        sys.exit(-1)

    # Save default glb file name and location using input gltf file name
    ext = 'glb'
    fname_out \
        = os.path.splitext(os.path.basename(input_glTF))[0] + '.' + ext

    if output_glb is not None:
        fname_out = output_glb
    else:
        fname_out = os.path.join(os.path.dirname(input_glTF), fname_out)

    with open(input_glTF, 'rb') as f:
        gltf = f.read()

    gltf = gltf.decode('utf-8')
    gltf_json = json.loads(gltf)

    # Set up body_encoder
    body_encoder \
        = BodyEncoder(containing_dir=os.path.dirname(input_glTF))

    buffer_map = {}
    buffer_offset = 0
    buffer_length = 0
    buffer_index = 0

    for i, _val in enumerate(gltf_json["buffers"]):
        try:
            length = gltf_json["buffers"][i]["byteLength"]
        except KeyError:
            length = None

        offset, length \
            = body_encoder.add_to_body(gltf_json["buffers"][i]["uri"], length)

        del gltf_json["buffers"][i]["uri"]
        gltf_json["buffers"][i]["byteLength"] = length

        buffer_index = i
        buffer_map[buffer_index] = buffer_offset
        buffer_offset += length

    # Iterate over the bufferViews to
    # move buffers into the single GLB buffer body
    for bufview in gltf_json["bufferViews"]:
        try:
            bufview["byteOffset"] \
                = bufview["byteOffset"] + buffer_map[bufview["buffer"]]
        except KeyError:
            bufview["byteOffset"] = 0

        bufview["buffer"] = 0

    # Iterate over images
    if 'images' in gltf_json:
        for i, image in enumerate(gltf_json["images"]):
            uri = image["uri"]
            if not uri:
                raise ValueError("GLTF image has empty URI.")
            buffer_map[buffer_index] = buffer_offset
            offset, length = body_encoder.add_to_body(uri, None)

            buffer_view = {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": length
            }

            # Update buffer_offset after add data into binary
            buffer_index += 1
            buffer_offset = offset
            buffer_length = length
            buffer_view_index = len(gltf_json["bufferViews"])
            gltf_json["bufferViews"].append(buffer_view)
            image["bufferView"] = buffer_view_index

            fileType, fileEncoding = mimetypes.guess_type(image["uri"])
            image["mimeType"] = fileType
            del image["uri"]

    #TODO: In our model pool, they doesn't contain shader resource.
    #if 'shaders' in gltf_json:

    gltf_json["buffers"][0]["byteLength"] = body_encoder.body_length
    lJSONStr = json.dumps(gltf_json, sort_keys=True, separators=(',', ':'))
    gltf_str = bytearray(lJSONStr.encode(encoding='UTF-8'))

    # Pack both binary json and data into single glb file
    encoder = GlbEncoder(gltf_str, body_encoder)
    encoder.export(fname_out)
