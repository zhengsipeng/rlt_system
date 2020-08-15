# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from .pascal_voc import pascal_voc
from .vidvrd import vidvrd
from .vidor import vidor
__sets = {}

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))


for year in ['2020']:
    for split in ['train', 'val', 'trainval', 'test', 'ext', 'coco']:
        name = 'vidvrd_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: vidvrd(split, year))
    for split in ['train', 'val', 'trainval', 'test', 'ext', 'coco', 'vrd', 'd1train', 'd2train', 'd1val', 'd2val']:
        name = 'vidor_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: vidor(split, year))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
