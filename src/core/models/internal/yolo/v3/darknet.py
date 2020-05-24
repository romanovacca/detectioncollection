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