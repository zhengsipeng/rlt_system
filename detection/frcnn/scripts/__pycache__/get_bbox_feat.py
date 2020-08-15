from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import sys
import argparse
import pprint
import time
import cv2
import pickle
import torch
import torch.nn as nn
import pickle as pkl
from tqdm import tqdm
from enum import Enum
from torch import Tensor
from torch.nn import functional as F
from frcnn.model.dataloader.roidb import combined_roidb
from frcnn.model.dataloader.roibatchLoader import roibatchLoader
from frcnn.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from frcnn.model.rpn.bbox_transform import clip_boxes
from frcnn.model.roi_layers import nms
from frcnn.model.rpn.bbox_transform import bbox_transform_inv
from frcnn.model.utils.net_utils import save_net, load_net, vis_detections
from frcnn.model.utils.blob import im_list_to_blob
from frcnn.model.faster_rcnn.resnet import resnet_base
from frcnn.model.faster_rcnn.resnet import resnet_fpn
from frcnn.model.roi_layers import ROIAlign
from frcnn.toolbox import *
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net', help='vgg16, res101', default='vgg16', type=str)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file',
                        default='frcnn/cfgs/res101.yml', type=str)
    parser.add_argument('--set', dest='set_cfgs', help='set config keys', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir', help='directory to load models', default="frcnn/models", type=str)
    parser.add_argument('--cuda', dest='cuda', help='whether use CUDA', action='store_true')
    parser.add_argument('--ls', dest='large_scale', help='whether use large imag scale', action='store_true')
    parser.add_argument('--bs', dest='batch_size', help='batch_size', default=4, type=int)
    parser.add_argument('--nw', dest='num_workers', help='number of worker to load data', default=0, type=int)
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--mGPUs', dest='mGPUs', help='whether use multiple GPUs', action='store_true')
    parser.add_argument('--cag', dest='class_agnostic', help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling', default=0,
                        type=int)
    parser.add_argument('--checksession', dest='checksession', help='checksession to load model', default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch', help='checkepoch to load network', default=20, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint to load network', default=20248, type=int)
    parser.add_argument('--split', dest='split', help='train, val or test', default='train', type=str)
    args = parser.parse_args()
    return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


class Pooler(object):
    class Mode(Enum):
        POOLING = 'pooling'
        ALIGN = 'align'
    OPTIONS = ['pooling', 'align']

    @staticmethod
    def apply(features: Tensor,
              proposal_bboxes: Tensor,
              proposal_batch_indices: Tensor,
              scale: Tensor,
              mode: Mode) -> Tensor:
        _, _, feature_map_height, feature_map_width = features.shape
        output_size = (7*2, 7*2)
        scale = 1. / scale

        if mode == Pooler.Mode.POOLING:
            pool = []
            for (proposal_bbox, proposal_batch_index) in zip(proposal_bboxes, proposal_batch_indices):
                start_x = max(min(torch.round(proposal_bbox[0].item() * scale).int() , feature_map_width - 1), 0)     # [0, feature_map_width)
                start_y = max(min(torch.round(proposal_bbox[1].item() * scale).int(), feature_map_height - 1), 0)    # (0, feature_map_height]
                end_x = max(min(torch.round(proposal_bbox[2].item() * scale).int()  + 1, feature_map_width), 1)        # [0, feature_map_width)
                end_y = max(min(torch.round(proposal_bbox[3].item() * scale).int() + 1, feature_map_height), 1)        # (0, feature_map_height]
                roi_feature_map = features[proposal_batch_index, :, start_y:end_y, start_x:end_x]
                pool.append(F.adaptive_max_pool2d(input=roi_feature_map, output_size=output_size))
            pool = torch.stack(pool, dim=0)
        elif mode == Pooler.Mode.ALIGN:
            pool = ROIAlign(output_size, spatial_scale=scale, sampling_ratio=0)(
                features,
                torch.cat([proposal_batch_indices.view(-1, 1).float(), proposal_bboxes], dim=1)
            )
        pool = F.max_pool2d(input=pool, kernel_size=2, stride=2)
        return pool


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def get_gt_feat(fasterRCNN, batch_list, split = 'test', dbname = 'mm19'):
    batch_count = -1
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    queue_dict = {}
    queue_vid, queue_sub = '', ''
    cap_sub, cap_vid = '', ''
    cap_frames = []

    feat_list = []
    for batch in tqdm(batch_list):
        batch_count += 1
        total_tic = time.time()
        blobs_list = []
        
        for im_file in batch: 
            sub = im_file.split('/')[0]
            vid = im_file.split('/')[1]
            fid = im_file.split('/')[2]
            if cap_vid != vid:
                cap_vid = vid
                cap_sub = sub
                cap_frames = []
                vpath = 'data/vidor/test/video/%s/%s.mp4' % (sub, vid)
                vcap = cv2.VideoCapture(vpath)
                frame_count = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
                cur_frame = 0
                frames = []
                if not vcap.isOpened():
                    raise Exception("cannot open %s" % vpath)
                copy_flag = False
                copy_num = 0
                while cur_frame < frame_count:
                    suc, frame = vcap.read()
                    if not suc:
                        copy_flag = True
                        copy_flag += 1
                    else:
                        if copy_flag:
                            if copy_num > 1:
                                print(copy_num, 'more than two adjacent frames fail')
                            for i in range(copy_num):
                                cap_frames.append(frame)
                            copy_flag = False
                            copy_num = 0
                        cap_frames.append(frame)
                    cur_frame += 1

            im_in = cap_frames[int(fid)]
            
            if len(im_in.shape) == 2:
                im_in = im_in[:, :, np.newaxis]
                im_in = np.concatenate((im_in, im_in, im_in), axis=2)
            
            blobs, im_scales = _get_image_blob(im_in)
            assert len(im_scales) == 1, "Only single-image batch implemented"
            im_blob = blobs
            blobs_list.append(im_blob)

        
        im_blobs = np.concatenate(blobs_list)
        im_info_np = np.array([[im_blobs.shape[1], im_blobs.shape[2], im_scales[0]]], dtype=np.float32)
        im_info_np = np.tile(im_info_np, (args.batch_size, 1))
        im_data_pt = torch.from_numpy(im_blobs)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)
        with torch.no_grad():
            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)

        feat_maps = fasterRCNN(im_data)  # scale = 1/16, [bs, c, h, w]

        for i, im_file in enumerate(batch):
            if i >= args.batch_size - pad_list[batch_count]:
                break
            bbox = [int(i) for i in im_file.split('/')[4:8]]
            sc = im_scales[0]

            # roi align
            rescale = np.asarray(bbox) * sc
            ind = torch.Tensor([0]).int().cuda()
            pooler = Pooler()
            mode = pooler.Mode('align')
            rois = []
            roi = torch.Tensor(rescale.reshape([1, 4])).cuda()  # [1, 4]   
            scale = torch.Tensor([16.]).float()  # the downsample scale = 16
            feat_map = feat_maps[i].unsqueeze(0)
            output = pooler.apply(feat_map, roi, ind, scale, mode)
            output = F.max_pool2d(input=output, kernel_size=7).cpu().detach().numpy()  # 7*7 -> 1*1
            rois.append(output) 
            rois = np.asarray(rois).reshape([1, 1024])
            feat_list.append(rois)

    with open('frcnn/cache/%s_rcnn_feat_%s.pkl'%(split, dbname), 'wb') as f:
        pkl.dump(feat_list, f)


def get_det_feat():
    batch_count = -1
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    queue_dict = {}
    queue_vid = ''
    queue_sub = ''

    all_boxes = [[[] for _ in range(num_images)]
                    for _ in range(81)]

    for batch in tqdm(batch_list):
        batch_count += 1
        total_tic = time.time()
        blobs_list = []
        
        # Detect 
        if not args.sum:
            vid_det_exist = False
            for im_file in batch: 
                sub = im_file.split('/')[0]
                vid = im_file.split('/')[1]
                fid = im_file.split('/')[2]
                pkl_file = 'frcnn/rcnn_feat/{:s}/{:s}.pkl'.format(sub, vid)
                if os.path.isfile(pkl_file):
                    vid_det_exist = True
                    break
                #print(im_file)
                impath = os.path.join(imgroot, sub, vid, '%05d.jpg'%int(fid))
                im_in = cv2.imread(impath)
                if len(im_in.shape) == 2:
                    im_in = im_in[:, :, np.newaxis]
                    im_in = np.concatenate((im_in, im_in, im_in), axis=2)
                # do not need to flip the channel, since the original one using cv2
                #im = im_in[:, :, ::-1]  
                
                blobs, im_scales = _get_image_blob(im_in)
                assert len(im_scales) == 1, "Only single-image batch implemented"
                im_blob = blobs
                blobs_list.append(im_blob)
            if vid_det_exist:
                continue
            #    #print('Dets already exist: output/res101/%s/%s/%s.pkl'%(args.imdbval_name, sub, vid))
            #    continue

            im_blobs = np.concatenate(blobs_list)

            im_info_np = np.array([[im_blobs.shape[1], im_blobs.shape[2], im_scales[0]]], dtype=np.float32)
            im_info_np = np.tile(im_info_np, (args.batch_size, 1))
            im_data_pt = torch.from_numpy(im_blobs)
            im_data_pt = im_data_pt.permute(0, 3, 1, 2)
            im_info_pt = torch.from_numpy(im_info_np)

            with torch.no_grad():
                im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
                im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
                gt_boxes.resize_(args.batch_size, 1, 5).zero_()
                num_boxes.resize_(args.batch_size).zero_()

            det_tic = time.time()
            rois, cls_prob, bbox_pred, rcnn_feat = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
            
            scores = cls_prob.data  # 81
            boxes = rois.data[:, :, 1:5]
            rcnn_feat = rcnn_feat.data
            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                    if args.class_agnostic:
                        if args.cuda > 0:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                        box_deltas = box_deltas.view(args.batch_size, -1, 4)
                    else:
                        if args.cuda > 0:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                        box_deltas = box_deltas.view(args.batch_size, -1, 4 * len(vidor_classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, args.batch_size)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, args.batch_size)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))
            
            pred_boxes /= im_scales[0]
            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            rcnn_feat = rcnn_feat.squeeze()
            det_toc = time.time()
            detect_time = det_toc - det_tic
            misc_tic = time.time()
        
        # Process
        for i, im_file in enumerate(batch):
            imid = i + startid_list[batch_count]
            if i >= args.batch_size - pad_list[batch_count]:
                break

            sub = im_file.split('/')[0]
            vid = im_file.split('/')[1]
            fid = im_file.split('/')[2]
            det_bbox = [int(coord) for coord in im_file.split('/')[-4:]]
            i_pred_boxes = pred_boxes[i]
            i_scores = scores[i]
            i_rcnn = rcnn_feat[i]
            bboxes = []
            for j in range(1, len(vidor_classes)):
                inds = torch.nonzero(i_scores[:, j] > thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = i_scores[:, j][inds]
                    all_scores = i_scores[:][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic:
                        cls_boxes = i_pred_boxes[inds, :]
                    else:
                        cls_boxes = i_pred_boxes[inds][:, j * 4:(j + 1) * 4]
                    inds_rcnn = i_rcnn[inds]
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    inds_rcnn = inds_rcnn[order]
                    all_scores = all_scores[order]
                    # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()].cpu()
                    inds_rcnn = inds_rcnn[keep.view(-1).long()].cpu()
                    all_scores = all_scores[keep.view(-1).long()].cpu()
                    clss = np.full([cls_dets.shape[0], 1], j)

                    for i in range(cls_dets.shape[0]):
                        print(det_bbox, cls_dets[i])
                        if det_bbox[0] == int(cls_dets[i][0]) and det_bbox[1] == int(cls_dets[i][1]) and det_bbox[2] == int(cls_dets[i][2]):
                            assert 1==0

                    bboxes.append(np.concatenate([cls_dets, clss, all_scores, inds_rcnn], axis=1))

            # write the detection results to file  
            if len(bboxes) == 0:
                print('NOne')
                bboxes = np.asarray([])
            else:
                bboxes = np.concatenate(bboxes, axis=0)
            if len(queue_dict.keys()) != 0 and vid != queue_vid:
                if not os.path.exists('frcnn/rcnn_feat/test/'+queue_sub):
                    os.makedirs('frcnn/rcnn_feat/test/'+queue_sub)
                pkl_file = 'frcnn/rcnn_feat/test/{:s}/{:s}.pkl'.format(queue_sub, queue_vid)
                if not os.path.isfile(pkl_file):
                    with open(pkl_file, 'wb') as f:
                        pkl.dump(queue_dict, f)
                else:
                    print('Already exist ', pkl_file)
                queue_dict = dict()  # empty the queue
            
            queue_dict[fid] = bboxes
            queue_vid = vid
            queue_sub = sub

    if not os.path.exists('frcnn/rcnn_feat/test/'+queue_sub):
        os.makedirs('frcnn/rcnn_feat/test/'+queue_sub)
    with open('frcnn/rcnn_feat/test/{:s}/{:s}.pkl'.format(queue_sub, queue_vid), 'wb') as f:
        pkl.dump(queue_dict, f)


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    cfg.USE_GPU_NMS = args.cuda

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)
    
    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
        'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    
    vidor_clss = get_vidor_clss()
    # initilize the network here.
    if args.net == 'res101':
        fasterRCNN = resnet_base(vidor_clss, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet_base(vidor_clss, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
    
    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    from collections import OrderedDict
    state_dict = checkpoint['model']
    new_state_dict = state_dict

    fasterRCNN.load_state_dict(new_state_dict)
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()


    if args.cuda > 0:
        cfg.CUDA = True
        fasterRCNN.cuda()
    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
    fasterRCNN.eval()
    
    start = time.time()
    max_per_image = 100
    thresh = 0.05
    vis = False

    with open('frcnn/cache/%s_%s_rcnn.txt'%(split, dbname), 'r') as f:
        imglist = [im.strip() for im in f.readlines()]
    
    num_images = len(imglist)
    print('Loaded Photo: {} images.'.format(num_images))

        # generate batch
    batch_list = []
    pad_list = []
    startid_list = []
    batch_imgs = []
    batch_vid = ''
    pad_len = 0
    inst_num = 0
    print('Generate batch')
    imid = 0
    while imid < num_images:  # 0 ~ num_images-1
        if imid % 100000 == 0:
            print('Remaining image num: {:d}'.format(imid))
        
        sub = imglist[imid].split('/')[0]
        vid = imglist[imid].split('/')[1]
        fid = imglist[imid].split('/')[2]
        if len(batch_imgs) == 0:  # the queue is empty
            # sub, vid, stid, otid,  0-3
            # ins_mid_fid, stt_fid, end_fid, rel_id, pre_cls,  4-8
            batch_imgs.append(imglist[imid])
            batch_vid = vid
            imid += 1  
            if imid == num_images:  # if it's the last image
                    pad_len = args.batch_size - len(batch_imgs)
                    inst_num += len(batch_imgs)
                    for i in range(pad_len):
                        batch_imgs.append(batch_imgs[-1])
                    batch_list.append(batch_imgs)
                    pad_list.append(pad_len)
        elif len(batch_imgs) < args.batch_size:
            if vid == batch_vid:
                batch_imgs.append(imglist[imid])
                imid += 1
                if imid == num_images:  # if it's the last image
                    pad_len = args.batch_size - len(batch_imgs)
                    inst_num += len(batch_imgs)
                    for i in range(pad_len):
                        batch_imgs.append(batch_imgs[-1])
                    batch_list.append(batch_imgs)
                    pad_list.append(pad_len)
            else:  # not equal , empty and do not +1
                pad_len = args.batch_size - len(batch_imgs)
                inst_num += len(batch_imgs)
                for i in range(pad_len):
                    batch_imgs.append(batch_imgs[-1])
                batch_list.append(batch_imgs)
                pad_list.append(pad_len)
                batch_imgs = []
                batch_vid = ''
                pad_len = 0
        else:
            batch_list.append(batch_imgs)
            inst_num += len(batch_imgs)
            pad_list.append(0)
            # initial the batch queue
            batch_imgs = []
            batch_vid = ''
            pad_len = 0
    print('Inst num: ', inst_num)
    data = {'batch_list': batch_list, 'pad_list': pad_list}
    assert len(batch_list) == len(pad_list)

    get_gt_feat(fasterRCNN, batch_list)


    