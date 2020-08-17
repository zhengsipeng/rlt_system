# --------------------------------------------------------
# This file is used for evaluate the val or test images and 
# get evaluation bboxes when batch_size > 1
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import sys
import argparse
import time
import cv2
import pickle
import torch
import torch.nn as nn
import pickle as pkl
import warnings
from tqdm import tqdm
from PIL import Image
from torch.autograd import Variable
from frcnn.model.dataloader.roidb import combined_roidb
from frcnn.model.dataloader.roibatchLoader import roibatchLoader
from frcnn.model.rpn.bbox_transform import clip_boxes, bbox_transform_inv
from frcnn.model.roi_layers import nms
from frcnn.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from frcnn.model.utils.net_utils import save_net, load_net, vis_detections
from frcnn.model.utils.blob import im_list_to_blob
from frcnn.model.faster_rcnn.resnet import resnet, resnet_fpn
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
    parser.add_argument('--bs', dest='batch_size', help='batch_size', default=8, type=int)
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
    parser.add_argument('--vis', dest='vis', help='visualization mode', action='store_true')
    parser.add_argument('--eval', dest='eval', default=False, type=str)
    parser.add_argument('--sum', dest='sum', help='sum det results of each vid together', default=False, type=str)
    args = parser.parse_args()
    return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


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

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    
    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
        'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    if 'd1' in args.dataset:
        # only evaluate human class
        vidor_classes = np.asarray(['__background__', 'adult', 'child', 'baby'])
        det_file = 'output/res101/faster_rcnn_10/detections_d1.pkl'
        with open('data/vidor/vidor_split/ImageSets/Main/val_d1.txt', 'r') as f:
            imglist = [im.strip() for im in f.readlines()]
        batch_list_file = 'output/res101/vald1_batch_list.pkl'
        args.imdbval_name = 'vidor_2020_d1val'
    elif 'd2' in args.dataset:
        # evaluate object
        vidor_classes = np.asarray(['__background__',
                            'bread', 'cake', 'dish', 'fruits', 'vegetables', 'backpack', 'camera',
                            'cellphone', 'handbag', 'laptop', 'suitcase', 'ball/sports_ball',
                            'bat', 'frisbee', 'racket', 'skateboard', 'ski', 'snowboard', 'surfboard', 
                            'baby_seat', 'bottle', 'chair', 'cup', 'electric_fan', 'faucet', 'microwave', 'oven',
                            'refrigerator', 'screen/monitor',
                            'sink', 'sofa', 'stool', 'table', 'toilet', 'guitar', 'piano', 'baby_walker', 'bench',
                            'stop_sign', 'traffic_light', 'aircraft', 'bicycle', 'bus/truck',
                            'car', 'motorcycle', 'scooter', 'train', 'watercraft', 'crab', 'bird', 'chicken',
                            'duck', 'penguin', 'fish', 'stingray', 'crocodile', 'snake', 'turtle', 'antelope',
                            'bear', 'camel', 'cat', 'cattle/cow',
                            'dog', 'elephant', 'hamster/rat',
                            'horse', 'kangaroo', 'leopard', 'lion', 'panda', 'pig', 'rabbit', 'sheep/goat',
                            'squirrel', 'tiger'
                            ])
        det_file = 'output/res101/faster_rcnn_10/detections_d2.pkl'
        with open('data/vidor/vidor_split/ImageSets/Main/val_d2.txt', 'r') as f:
            imglist = [im.strip() for im in f.readlines()]
        batch_list_file = 'output/res101/vald2_batch_list.pkl'
        args.imdbval_name = 'vidor_2020_d2val'
    else:
        # 80 classes evaluation
        vidor_classes = np.asarray(['__background__',
                            'bread', 'cake', 'dish', 'fruits', 'vegetables', 'backpack', 'camera',
                            'cellphone', 'handbag', 'laptop', 'suitcase', 'ball/sports_ball',
                            'bat', 'frisbee', 'racket', 'skateboard', 'ski', 'snowboard', 'surfboard', 'toy',
                            'baby_seat', 'bottle', 'chair', 'cup', 'electric_fan', 'faucet', 'microwave', 'oven',
                            'refrigerator', 'screen/monitor',
                            'sink', 'sofa', 'stool', 'table', 'toilet', 'guitar', 'piano', 'baby_walker', 'bench',
                            'stop_sign', 'traffic_light', 'aircraft', 'bicycle', 'bus/truck',
                            'car', 'motorcycle', 'scooter', 'train', 'watercraft', 'crab', 'bird', 'chicken',
                            'duck', 'penguin', 'fish', 'stingray', 'crocodile', 'snake', 'turtle', 'antelope',
                            'bear', 'camel', 'cat', 'cattle/cow',
                            'dog', 'elephant', 'hamster/rat',
                            'horse', 'kangaroo', 'leopard', 'lion', 'panda', 'pig', 'rabbit', 'sheep/goat',
                            'squirrel', 'tiger', 'adult', 'baby', 'child'
                            ])
        
        if not 'test' in args.dataset:
            # validation set
            det_file = 'output/res101/vidor_2020_val/faster_rcnn_10/detections.pkl'
            with open('data/vidor/ImageSets/Main/val.txt', 'r') as f:
                imglist = [im.strip() for im in f.readlines()]
            batch_list_file = 'output/res101/val_batch_list.pkl'
            args.imdbval_name = 'vidor_2020_val'
        else:
            # test set
            # there are 3 position need to be correct
            det_file = 'output/res101/vidor_2020_test/faster_rcnn_10/detections.pkl'
            with open('data/vidor/test/test.txt', 'r') as f:
                imglist = [im.strip() for im in f.readlines()]
            batch_list_file = 'output/res101/test_batch_list.pkl'
            args.imdbval_name = 'vidor_2020_test'

    if not args.eval:
        if not args.sum:
            # initilize the network here.
            if args.net == 'res101':
                fasterRCNN = resnet(vidor_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
            elif args.net == 'res152':
                fasterRCNN = resnet(vidor_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
            else:
                print("network is not defined")
                pdb.set_trace()
            
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


            print('load model successfully!')

            print("load checkpoint %s" % (load_name))        
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
        
        num_images = len(imglist)
        print('Loaded Photo: {} images.'.format(num_images))

        # generate batch
        batch_list = []
        pad_list = []
        startid_list = []
        batch_imgs = []
        batch_vid = ''
        pad_len = 0
        print('Generate batch')
            
        if os.path.exists(batch_list_file):
            print('loading batch_list from batch_list_file')
            with open(batch_list_file, 'rb') as f:
                data = pkl.load(f)
                batch_list = data['batch_list']
                pad_list = data['pad_list']
                startid_list = data['startid_list']
        else:
            imid = 0
            if 'test' in args.dataset:
                imgroot = 'data/vidor/test/JPEGImages'
            else:
                imgroot = 'data/vidor/JPEGImages'
            while imid < num_images:  # 0 ~ num_images-1
                if imid % 100000 == 0:
                    print('Remaining image num: {:d}'.format(imid))
                if len(batch_imgs) == 0:  # the queue is empty
                    im_file = os.path.join(imgroot, imglist[imid].split('-')[0],
                                          imglist[imid].split('-')[1], imglist[imid].split('-')[2]+'.jpg')
                    batch_imgs.append(im_file)
                    batch_vid = imglist[imid].split('-')[1]
                    startid_list.append(imid)
                    imid += 1  
                    if imid == num_images:  # if it's the last image
                            pad_len = args.batch_size - len(batch_imgs)
                            for i in range(pad_len):
                                batch_imgs.append(batch_imgs[-1])
                            batch_list.append(batch_imgs)
                            pad_list.append(pad_len)
                elif len(batch_imgs) < args.batch_size:
                    vid = imglist[imid].split('-')[1]
                    if vid == batch_vid:
                        im_file = os.path.join(imgroot, imglist[imid].split('-')[0],
                                          imglist[imid].split('-')[1], imglist[imid].split('-')[2]+'.jpg')
                        batch_imgs.append(im_file)
                        imid += 1
                        if imid == num_images:  # if it's the last image
                            pad_len = args.batch_size - len(batch_imgs)
                            for i in range(pad_len):
                                batch_imgs.append(batch_imgs[-1])
                            batch_list.append(batch_imgs)
                            pad_list.append(pad_len)
                    else:  # not equal , empty and do not +1
                        pad_len = args.batch_size - len(batch_imgs)
                        for i in range(pad_len):
                            batch_imgs.append(batch_imgs[-1])
                        batch_list.append(batch_imgs)
                        pad_list.append(pad_len)
                        batch_imgs = []
                        batch_vid = ''
                        pad_len = 0
                else:
                    batch_list.append(batch_imgs)
                    pad_list.append(0)
                    # initial the batch queue
                    batch_imgs = []
                    batch_vid = ''
                    pad_len = 0

            data = {'batch_list': batch_list, 'pad_list': pad_list, 'startid_list': startid_list}
            with open(batch_list_file, 'wb') as f:
                    pkl.dump(data, f)

        assert len(batch_list) == len(pad_list)
        assert len(batch_list) == len(startid_list)

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
                    sub = im_file.split('/')[-3]
                    vid = im_file.split('/')[-2]
                    fid = im_file.split('/')[-1]
                    pkl_file = 'output/res101/{:s}/{:s}/{:s}.pkl'.format(args.imdbval_name, sub, vid)
                    if os.path.isfile(pkl_file):
                        vid_det_exist = True
                        break
 
                    im_in = cv2.imread(im_file)
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
                #try:
                im_blobs = np.concatenate(blobs_list)
                #except:
                #    print(batch)
                #    assert 1 == 0
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
                rois, cls_prob, bbox_pred, _ = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
                
                scores = cls_prob.data
                boxes = rois.data[:, :, 1:5]
            
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
                
                det_toc = time.time()
                detect_time = det_toc - det_tic
                misc_tic = time.time()
            
            # Process
            for i, im_file in enumerate(batch):
                imid = i + startid_list[batch_count]
                if i >= args.batch_size - pad_list[batch_count]:
                    break

                sub = im_file.split('/')[-3]
                vid = im_file.split('/')[-2]
                fid = im_file.split('/')[-1]
                
                if args.sum:
                    if vid != queue_vid:
                        with open('output/res101/%s/%s/%s.pkl'%(args.imdbval_name, sub, vid), 'rb') as f:
                            queue_dict = pkl.load(f)
                        queue_sub, queue_vid = sub, vid
                    else:
                        if fid not in queue_dict:
                            print(vid, fid)
                            continue
                        bboxes = queue_dict[fid]
                        if bboxes.shape[0] == 0:
                            continue
                        for j in range(1, len(vidor_classes)):
                            #print(np.nonzero(bboxes[:, 4] == j))
                            inds = np.nonzero(bboxes[:, 5] == j)[0]
                            if inds.shape[0] > 0:
                                cls_dets = bboxes[inds, :4]
                                all_boxes[j][imid] = cls_dets
                    continue
                
                i_pred_boxes = pred_boxes[i]
                i_scores = scores[i]

                bboxes = []
                for j in range(1, len(vidor_classes)):
                    inds = torch.nonzero(i_scores[:, j] > thresh).view(-1)
                    # if there is det
                    if inds.numel() > 0:
                        cls_scores = i_scores[:, j][inds]
                        _, order = torch.sort(cls_scores, 0, True)
                        if args.class_agnostic:
                            cls_boxes = i_pred_boxes[inds, :]
                        else:
                            cls_boxes = i_pred_boxes[inds][:, j * 4:(j + 1) * 4]
                        
                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                        # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                        cls_dets = cls_dets[order]
                        # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
                        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                        cls_dets = cls_dets[keep.view(-1).long()].cpu()
                        
                        clss = np.full([cls_dets.shape[0], 1], j)
                        bboxes.append(np.concatenate([cls_dets, clss], axis=1))

                        all_boxes[j][imid] = cls_dets.cpu().numpy()
                        if vis:
                            im2show = vis_detections(cv2.imread(im_file), vidor_classes[j], cls_dets, 0.05)
                            result_path = os.path.join('data/vis/vidor_val', 
                                                    vid+'-'+str(j)+'-'+fid)
                            cv2.imwrite(result_path, im2show)
                    else:
                        all_boxes[j][imid] = empty_array

                # Limit to max_per_image detections *over all classes*
                if max_per_image > 0:
                    image_scores = np.hstack([all_boxes[j][imid][:, -1] for j in range(1, len(vidor_classes))])
                    if len(image_scores) > max_per_image:
                        image_thresh = np.sort(image_scores)[-max_per_image]
                        for j in range(1, len(vidor_classes)):
                            keep = np.where(all_boxes[j][imid][:, -1] >= image_thresh)[0]
                            
                            all_boxes[j][imid] = all_boxes[j][imid][keep, :]


                 # write the detection results to file  
                if len(bboxes) == 0:
                    bboxes = np.asarray([])
                else:
                    bboxes = np.concatenate(bboxes, axis=0)
                if len(queue_dict.keys()) != 0 and vid != queue_vid:
                    if not os.path.exists('output/res101/%s/'%args.imdbval_name+queue_sub):
                        os.makedirs('output/res101/%s/'%args.imdbval_name+queue_sub)
                    pkl_file = 'output/res101/{:s}/{:s}/{:s}.pkl'.format(args.imdbval_name, queue_sub, queue_vid)
                    if not os.path.isfile(pkl_file):
                        with open(pkl_file, 'wb') as f:
                            pkl.dump(queue_dict, f)
                    else:
                        print('Already exist ', pkl_file)
                    queue_dict = dict()  # empty the queue
                queue_dict[fid] = bboxes
                queue_vid = vid
                queue_sub = sub

        if not os.path.exists('output/res101/%s/'%args.imdbval_name+queue_sub):
            os.makedirs('output/res101/%s/'%args.imdbval_name+queue_sub)
        with open('output/res101/{:s}/{:s}/{:s}.pkl'.format(args.imdbval_name, queue_sub, queue_vid), 'wb') as f:
            pkl.dump(queue_dict, f)
        
        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
            
    else:
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
        imdb.competition_mode(on=True)
        print('Loading detection results')
        with open(det_file, 'rb') as f:
            all_boxes = pickle.load(f)
        print('Detection loaded')

        print('Evaluating detections')
        output_dir = get_output_dir(imdb, 'faster_rcnn_10')
        print(output_dir)
        imdb.evaluate_detections(all_boxes, output_dir)

        end = time.time()
        print("test time: %0.4fs" % (end - start))
