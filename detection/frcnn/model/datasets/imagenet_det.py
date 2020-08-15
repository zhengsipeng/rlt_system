from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import scipy.sparse
import uuid
import pickle as pkl
import scipy.io as sio
import xml.etree.ElementTree as ET
from . import ds_utils
from .imdb import imdb
from .vrd_eval import vrd_eval
from frcnn.model.utils.config import cfg


class imagenet_det(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'imagenet_det_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = cfg.DATA_DIR+'/imagenet_det'
        self._data_path = cfg.DATA_DIR+'/imagenet_det'