from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import numpy as np
import scipy.sparse
import uuid
import pickle as pkl
import scipy.io as sio
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from tqdm import tqdm
from . import ds_utils
from .imdb import imdb
from .vrd_eval import vrd_eval
from frcnn.model.utils.config import cfg


class vidvrd(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'vidvrd_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set  # vidvrd_2020_ext use imagenet_det to extend the dataset
        if self._image_set == 'ext':
            self._data_path = cfg.DATA_DIR+'/imagenet_det/ILSVRC2015'
        elif self._image_set == 'coco':
            self._data_path = cfg.DATA_DIR+'/coco'
        else:
            self._data_path = cfg.DATA_DIR+'/vidvrd'
        self._classes = ('__background__',  # always index 0
                          "turtle", "antelope", "bicycle", "lion", "ball", "motorcycle", "cattle", "airplane", "red_panda",
                          "horse", "watercraft", "monkey", "fox", "elephant", "bird", "sheep", "frisbee", "giant_panda",
                         "squirrel", "bus", "bear", "tiger", "train", "snake", "rabbit", "whale", "sofa", "skateboard", "dog",
                         "domestic_cat", "person", "lizard", "hamster", "car", "zebra")

        self.clscode2clsname = {
             'n01662784': 'turtle', 'n02419796': 'antelope', 'n02834778': 'bicycle', 'n02129165': 'lion',
             'n02691156': 'airplane', 'n02802426': 'ball', 'n03134739': 'ball', 'n03445777': 'ball',
             'n03942813': 'ball', 'n04118538': 'ball', 'n02799071': 'ball', 'n04254680': 'ball',
             'n04409515': 'ball', 'n04540053': 'ball', 'n03790512': 'motorcycle', 'n02402425': 'cattle',
             'n02509815': 'red_panda', 'n02374451': 'horse', 'n04530566': 'watercraft', 'n02484322': 'monkey',
             'n02118333': 'fox', 'n02503517': 'elephant', 'n01503061': 'bird', 'n02411705': 'sheep',
             'n02510455': 'giant_panda', 'n02355227': 'squirrel', 'n02924116': 'bus', 'n02131653': 'bear',
             'n02129604': 'tiger', 'n04468005': 'train', 'n01726692': 'snake', 'n02324045': 'rabbit',
             'n02062744': 'whale', 'n04256520': 'sofa', 'n02084071': 'dog', 'n02121808': 'domestic_cat',
             'n00007846': 'person', 'n01674464': 'lizard', 'n02342885': 'hamster', 'n02958343': 'car',
             'n02391049': 'zebra'
            }
        print('Number of classes: {}'.format(self.num_classes))

        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        #print(self._class_to_ind)
        if self._image_set == 'ext':
            self._image_ext = '.JPEG'
        else:
            self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # Dataset specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'top_k': 2000,
                       'use_diff': False,
                       'rpn_file': None}
        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        if self._image_set == 'ext':
            raw_path = index.split(' ')[0]  # ILSVRC2013_train_extra0/ILSVRC2013_train_00000001
            year_set = raw_path.split('/')[0].split('_')[0]
            if '2013' in year_set:
                subdir = raw_path.split('/')[0]
                clscode = raw_path.split('/')[1]
                frameid = raw_path.split('/')[2]
                image_path = os.path.join(self._data_path, 'Data', 'DET', 'train', subdir, clscode,
                                          frameid+self._image_ext)
            else:
                subdir = raw_path.split('/')[0]
                frameid =raw_path.split('/')[1]
                image_path = os.path.join(self._data_path, 'Data', 'DET', 'train', subdir, frameid+self._image_ext)
        elif self._image_set == 'coco':
            image_path = os.path.join(self._data_path, 'train2017', index+self._image_ext)
        else:
            vidname = index.split('-')[0]
            frameid = index.split('-')[1]
            image_path = os.path.join(self._data_path, 'JPEGImages', vidname, frameid+self._image_ext)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        if self._image_set == 'ext':
            image_set_index_file = os.path.join(self._data_path, 'ImageSets/DET/train_vidvrd.txt')
        elif self._image_set == 'coco':
            image_set_index_file = os.path.join(self._data_path, 'ImageSets/Main/train_vidvrd.txt')
        else:
            image_set_index_file = os.path.join(self._data_path, 'ImageSets/Main', self._image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pkl.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb
        gt_roidb = []
        tqdm_image_index = tqdm(self.image_index)
        tqdm_image_index.set_description('gt_roidb_loading')
        for index in tqdm_image_index:
            gt_roidb.append(self._load_vidvrd_annotation(index))
        with open(cache_file, 'wb') as fid:
            pkl.dump(gt_roidb, fid, pkl.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pkl.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            pkl.dump(roidb, fid, pkl.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))

        return roidb

    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)
        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pkl.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in range(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_vidvrd_annotation(self, index):

        """ given index, load image and bounding boxes info from XML file
        """
        if self._image_set == 'ext':
            filename = os.path.join(self._data_path, 'Annotations', 'DET', 'train_vidvrd', index.split(' ')[0]+'.xml')
        elif self._image_set == 'coco':
            filename = os.path.join(self._data_path, 'Annotations', 'train_vidvrd', index+'.xml')
        else:
            video_name = index.split('-')[0]
            frame_id = index.split('-')[1]
            filename = os.path.join(self._data_path, 'Annotations', self._image_set,
                                    video_name, str(int(frame_id))+'.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        width = float(tree.find('size').find('width').text)
        height = float(tree.find('size').find('height').text)
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        valid_objs = np.zeros((num_objs), dtype=np.bool)
        class_to_index = dict(zip(self._classes, range(self.num_classes)))

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = max(min(float(bbox.find('xmin').text) - 1, width - 1), 0)
            y1 = max(min(float(bbox.find('ymin').text) - 1, height - 1), 0)
            x2 = max(min(float(bbox.find('xmax').text) - 1, width - 1), 0)
            y2 = max(min(float(bbox.find('ymax').text) - 1, height - 1), 0)
            if self._image_set == 'ext':
                if not obj.find('name').text in self.clscode2clsname.keys():
                    continue
                cls = class_to_index[self.clscode2clsname[obj.find('name').text.lower().strip()]]
            else:
                if not obj.find('name').text in class_to_index.keys():
                    continue
                cls = class_to_index[obj.find('name').text.lower().strip()]
            valid_objs[ix] = True
            boxes[ix, :] = [x1, y1, x2, y2]

            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        boxes = boxes[valid_objs, :]
        gt_classes = gt_classes[valid_objs]
        overlaps = overlaps[valid_objs, :]
        overlaps = scipy.sparse.csr_matrix(overlaps)
        assert (boxes[:, 2] >= boxes[:, 0]).all()

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_vidvrd_results_file_template(self):
        # devkit/results/det_test_aeroplane.txt
        filename = 'det_' + self._image_set + '_{:s}.txt'
        base_path = os.path.join(self._data_path, 'results')
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        path = os.path.join(self._data_path, 'results', 'vidvrd/Main/')
        if not os.path.exists(path):
            os.makedirs(path)
        return os.path.join(path, filename)

    def _write_vidvrd_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} vidvrd results file'.format(cls))
            filename = self._get_vidvrd_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self._image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def save_track_xml(self, all_boxes):
        img_num = len(self._image_index)
        for im_ind, index in enumerate(self._image_index):
            # get img size for anno 0.xml
            tree = ET.parse('data/vidvrd/Annotations/test/{:s}/0.xml'.format(index.split('-')[0]))
            iw = int(tree.find('size').find('width').text)
            ih = int(tree.find('size').find('height').text)

            viddir = 'data/vidvrd/track_bbox/{:s}/'.format(index.split('-')[0])
            if not os.path.exists(viddir):
                os.makedirs(viddir)
            fid = int(index.split('-')[1])

            xml_name = os.path.join(viddir, str(fid) + '.xml')
            bboxes, clss = [], []
            for cls_ind, cls in enumerate(self.classes):     
                # index ILSVRC2015_val_00051001-00177
                dets = all_boxes[cls_ind][im_ind]
                for det in dets:
                    if det[4] < 0.001:
                        continue
                    bboxes.append([det[0], det[1], det[2], det[3], det[4]])
                    clss.append(cls)
            dom = self.make_xml(fid, ih, iw, bboxes, clss)
            with open(xml_name, 'wb') as f:
                f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))
            
            sys.stdout.write('im_detect: {:d}/{:d}   \r' \
                         .format(im_ind, img_num))
            sys.stdout.flush()

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(self._data_path, 'Annotations', self._image_set, '{:s}', '{:d}.xml')
        imagesetfile = os.path.join(self._data_path, 'ImageSets', 'Main', self._image_set + '.txt')
        cachedir = os.path.join(self._data_path, 'annotations_cache')
        aps = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_vidvrd_results_file_template().format(cls)
            rec, prec, ap = vrd_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pkl.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, all_boxes, output_dir):
        # self._image_index = ['/'.join(roi_entry[0]['image'].split('/')[-3:])\
        #                        .replace('.JPEG','').replace('.jpeg', '')\
        #                        .replace('.jpg','').replace('.JPG','') \
        #                        for roi_entry in self._roidb]
        #self._write_vidvrd_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_vidvrd_results_file_template().format(cls)
                os.remove(filename)

    #def get_track
    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

    def make_xml(self, im_id, ih, iw, boxes, clss):
        node_root = Element('annotation')
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = str(im_id)
        node_source = SubElement(node_root, 'source')
        node_dataset = SubElement(node_source, 'dataset')
        node_dataset.text = 'vidvrd'

        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = str(iw)
        node_height = SubElement(node_size, 'height')
        node_height.text = str(ih)
        node_depth = SubElement(node_size, 'depth')
        node_depth.text = '3'

        for i, box in enumerate(boxes):
            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            node_name.text = clss[i]

            xmin, ymin, xmax, ymax, conf = round(box[0], 3), round(box[1], 3), round(box[2], 3), \
                                           round(box[3], 3), round(box[4], 3)

            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = str(xmin)
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(ymin)
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(xmax)
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(ymax)
            node_conf = SubElement(node_bndbox, 'confidence')
            node_conf.text = str(conf)

        xml = tostring(node_root)
        dom = parseString(xml)
        return dom