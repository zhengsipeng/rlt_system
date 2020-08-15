import os
import warnings
import argparse
from tqdm import tqdm
import torch
from frcnn.model.dataloader.roibatchLoader import roibatchLoader
from frcnn.model.roi_layers import nms
from frcnn.model.dataloader.roidb import combined_roidb
from frcnn.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from frcnn.toolbox import *
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='vidor', type=str)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file',
                        default='frcnn/cfgs/res101.yml', type=str)
    parser.add_argument('--set', dest='set_cfgs', help='set config keys', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir', help='directory to load models', default="frcnn/models", type=str)
    parser.add_argument('--cuda', dest='cuda', help='whether use CUDA', action='store_true')
    parser.add_argument('--cag', dest='class_agnostic', help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling', default=0,
                        type=int)
    parser.add_argument('--vis', dest='vis', help='visualization mode', action='store_true')
    parser.add_argument('--val', dest='val', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(cfg.RNG_SEED)
    split = 'test'

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == 'vidvrd':
        args.imdb_name = 'vidvrd_2020_train+vidvrd_2020_ext+vidvrd_2020_coco'
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        args.imdbval_name = 'vidvrd_2020_{:s}'.format(split)
    elif args.dataset == 'vidor':
        args.imdb_name = 'vidor_2020_train+vidor_2020_ext+vidor_2020_coco'
        args.imdbval_name = 'vidor_2020_{:s}'.format(split)
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.dataset[:5] != 'vidor':
        cfg.TRAIN.USE_FLIPPED = False
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
        imdb.competition_mode(on=True)

        print('{:d} roidb entries'.format(len(roidb)))
        output_dir = get_output_dir(imdb, 'faster_rcnn_10')
        print(output_dir)
        det_file = os.path.join(output_dir, 'detections.pkl')
        with open(det_file, 'rb') as f:
            all_boxes = pickle.load(f)

        print('Generate bbox for tracking')
        imdb.save_traj_xml(all_boxes)
    else:
        dbname = args.dataset[:5]
        print('Loading imglist')
        with open('data/%s/ImageSets/Main/val.txt'%dbname, 'r') as f:
            imglist = [im.strip() for im in f.readlines()]
        print('Loading detection results')
        with open('output/res101/save/%s_2020_val_epo20/faster_rcnn_10/detections.pkl'%dbname, 'rb') as f:
            all_boxes = pickle.load(f)
        det_to_xml()(all_boxes, imglist)

