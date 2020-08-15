###############################################################
#                   /   _oooo000oooo_   \
#                  /   o8888888888888o   \
#                  |   88"    .    "88   |
#                  |   (| ^   _   ^ |)   |
#                  \   0\     =     /0   /
#                   `___/`_________'\___'
#                 .'  \\|           |//  `.
#                /  \\|||     :     |||\\  \
#               /   _||||    -:-    ||||_   \
#               |    | \\\    -    /// |    | 
#               |  \_|    ''\---/''    |_/  |
#               \  .-\__     `_`     __/-.  /
#             ___`. .'     /--.--\    '. .  ___
#           ."" '<    `.___\_<|>_/___.'    >' "".
#         |  | :   `- \`.;` \ _ / ';.`/ -`   : |  |
#         \  \  `_.    \_ __ \ / __ _/    ._'  /  /
# =========`-.______`-.____ \___/ ____.-'_____.-'==============   
#                          `=~~~='
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#          Buddha, bless me keeping away from bugs
###############################################################
import os
import argparse
import pprint
import pdb
import time
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from frcnn.model.utils.config import cfg, cfg_from_file, cfg_from_list
from frcnn.model.dataloader.roidb import combined_roidb
from frcnn.model.dataloader.sampler import sampler
from frcnn.model.dataloader.roibatchLoader import roibatchLoader
from frcnn.model.faster_rcnn.resnet import resnet
from frcnn.model.faster_rcnn.resnet import resnet_fpn
from frcnn.model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net', help='vgg16, res101', default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch', help='starting epoch', default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs', help='number of epochs to train', default=30, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval', help='number of iterations to display', default=100,
                        type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval', help='number of iterations to display',
                        default=10000, type=int)
    parser.add_argument('--save_dir', dest='save_dir', help='directory to save model', default="models", type=str)
    parser.add_argument('--nw', dest='num_workers', help='number of worker to load data', default=0, type=int)
    parser.add_argument('--cuda', dest='cuda', help='whether use CUDA', action='store_true')
    parser.add_argument('--ls', dest='large_scale', help='whether use large imag scale', action='store_true')
    parser.add_argument('--fpn', dest='fpn', help='use fpn or not', action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs', help='whether use multiple GPUs', action='store_true')
    parser.add_argument('--bs', dest='batch_size', help='batch_size', default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic', help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer', help='training optimizer', default="sgd", type=str)
    parser.add_argument('--lr', dest='lr', help='starting learning rate', default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step', help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session', help='training session', default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume', help='resume checkpoint or not', default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession', help='checksession to load model', default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch', help='checkepoch to load model', default=6, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint to load model', default=221230, type=int)

    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard', help='whether use tensorboard', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.cfg_file = "frcnn/cfgs/{}_ls.yml".format(args.net) if args.large_scale \
        else "frcnn/cfgs/{}.yml".format(args.net)
    print('Called with args:')
    print(args)

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == 'vidvrd':
        args.imdb_name = 'vidvrd_2020_train'
        args.imdbval_name = 'vidvrd_2020_test'
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == 'vidor':
        args.imdb_name = 'vidor_2020_train+vidor_2020_ext+vidor_2020_coco+vidor_2020_vrd'
        #args.imdb_name = 'vidor_2020_ext'
        args.imdbval_name = 'vidor_2020_val'
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == 'vidor_d1':
        # train the subset 'adult, child, baby, toy' of vidor
        args.imdb_name = 'vidor_2020_d1train'
        args.imdbval_name = 'vidor_2020_d2val'
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == 'vidor_d2':
        args.imdb_name = 'vidor_2020_d2train+vidor_2020_ext+vidor_2020_coco+vidor_2020_vrd'
        args.imdbval_name = 'vidor_2020_d2val'
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    if args.fpn:
        if args.dataset == "coco":
            args.set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                             'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == 'vidvrd':
            args.set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                             'MAX_NUM_GT_BOXES', '50']

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = False
    cfg.USE_GPU_NMS = args.cuda
    cfg.CUDA = True

    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    print('{:d} roidb entries'.format(len(roidb)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sampler_batch = sampler(train_size, args.batch_size)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, imdb.num_classes, training=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             sampler=sampler_batch, num_workers=args.num_workers, 
                                             shuffle=False)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    im_data = im_data.cuda()

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # initilize the network here.
    if args.net == 'res101':
        if args.fpn:
            fasterRCNN = resnet_fpn(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
        else:
            fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        if args.fpn:
            fasterRCNN = resnet_fpn(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
        else:
            fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        raise NotImplementedError

    fasterRCNN.create_architecture()
    
    #lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
    fasterRCNN.cuda()

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM, lr=lr)

    if args.resume:
        load_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, \
                                 args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    iters_per_epoch = int(train_size / args.batch_size)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs")

    # training procedure
    for epoch in range(args.start_epoch, args.max_epochs + 1):  #
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)

        for step in range(iters_per_epoch):     
            data = next(data_iter)
            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])

            fasterRCNN.zero_grad()

            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)
                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                                       (epoch - 1) * iters_per_epoch + step)

                loss_temp = 0
                start = time.time()
 
        save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

    if args.use_tfboard:
        logger.close()




