from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def parse_cfg(cfgfile):
    """ Takes a configuration file.

    These blocks are sections from the configuratin file.
    Every block can be considered a layer from the network.
    A block is represented as a dictionary in the list that this function returns.

    """

    file = open(cfgfile, 'r')
    # store the lines in a list
    lines = file.read().split('\n')
    # filter out the empty lines
    lines = [x for x in lines if len(x) > 0]
    # filter out the comment lines
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    blocks = _parse_cfg(lines)
    return blocks


def _parse_cfg(lines):
    """ creates the blocks from the config file

    It uses a placeholder variable that stores all the values from one block until the next block is detected.
    Then it writes the completed block to the block list, cleans the placeholder block and fills it with the next block
    and repeats the cycle.
    """
    block = {}
    blocks = []

    for line in lines:
        # "[" marks the start of a new block
        if line[0] == "[":
            #
            # If block is not empty, implies it is storing values of previous block.
            if len(block) != 0:
                # add it the blocks list
                blocks.append(block)
                # re-init the block
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def create_pytorch_modules(blocks):
    """Construct PyTorch modules for the blocks present in the config file.

     PyTorch provides pre-built layers for types "convolutional" and "upsample".
     We will have to write our own modules for the rest of the layers by extending the nn.Module class.
    """
    # Captures the information about the input and pre-processing such as batch size, image size, gamma etc.
    net_info = blocks.pop(0)

    module_list = nn.ModuleList()

    # Keep track of number of filters in the layer on which the convolutional layer is being applied on.
    # During initialization this is 3 as the image has 3 filters corresponding to the RGB channels
    prev_filters = 3
    # keep a track of all preceding filters
    output_filters = []

    for index, x in enumerate(blocks):
        module = nn.Sequential()

        # check the type of block, create a new module for the block, append to module_list

        # If it's a convolutional layer
        if (x["type"] == "convolutional"):
            # Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

            # If it's an upsampling layer
            # We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)

        # If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            # Start  of a route
            start = int(x["layers"][0])
            # end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            # Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        # Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
