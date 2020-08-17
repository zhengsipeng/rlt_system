import cv2
import os
import json
import numpy as np
import argparse
import random
import pickle as pkl
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from tqdm import tqdm
from PIL import Image
parser = argparse.ArgumentParser('Data_Process')
parser.add_argument('--func', dest='func', default='tmp', type=str)
parser.add_argument('--imds', dest='imds', default='vidor', type=str)
parser.add_argument('--frame_gap', dest='frame_gap', default=8, type=int)
args = parser.parse_args()

global modes
if args.imds == 'vidvrd':
    modes = ['train', 'test']
elif args.imds == 'vidor':
    modes = ['train', 'val']
else:
    raise NotImplementedError


# ------------------------------
# statistic and data analysis
# ------------------------------
global obj_cls2id
global rel_cls2id
if args.imds == 'vidvrd':
    obj_num = 35
    rel_num = 132
    obj_cls2id = {
        "turtle": 0, "antelope": 1, "bicycle": 2, "lion": 3, "ball": 4, "motorcycle": 5, "cattle": 6, "airplane": 7,
        "red_panda": 8, "horse": 9, "watercraft": 10, "monkey": 11, "fox": 12, "elephant": 13, "bird": 14, "sheep": 15,
        "frisbee": 16, "giant_panda": 17, "squirrel": 18, "bus": 19, "bear": 20, "tiger": 21, "train": 22, "snake": 23,
        "rabbit": 24, "whale": 25, "sofa": 26, "skateboard": 27, "dog": 28, "domestic_cat": 29, "person": 30,
        "lizard": 31,
        "hamster": 32, "car": 33, "zebra": 34
    }
elif args.imds == 'vidor':
    obj_num = 80
    rel_num = 50
    vidor_clss = ['adult', 'aircraft', 'antelope', 'baby', 'baby_seat', 'baby_walker', 'backpack', 'ball/sports_ball',
                  'bat', 'bear', 'bench', 'bicycle', 'bird', 'bottle', 'bread', 'bus/truck',
                  'cake', 'camel', 'camera', 'car', 'cat', 'cattle/cow', 'cellphone', 'chair',
                  'chicken', 'child', 'crab', 'crocodile', 'cup', 'dish', 'dog', 'duck',
                  'electric_fan', 'elephant', 'faucet', 'fish', 'frisbee', 'fruit', 'guitar', 'hamster/rat',
                  'handbag', 'horse', 'kangaroo', 'laptop', 'leopard', 'lion', 'microwave', 'motorcycle',
                  'oven', 'panda', 'penguin', 'piano', 'pig', 'rabbit', 'racket', 'refrigerator',
                  'scooter', 'screen/monitor', 'sheep/goat', 'sink', 'skateboard', 'ski', 'snake', 'snowboard',
                  'sofa', 'squirrel', 'stingray', 'stool', 'stop_sign', 'suitcase', 'surfboard', 'table',
                  'tiger', 'toilet', 'toy', 'traffic_light', 'train', 'turtle', 'vegetables', 'watercraft']
    obj_cls2id = {}
    for i, cls in enumerate(vidor_clss):
        obj_cls2id[cls] = i
else:
    raise NotImplementedError
obj_id2cls = dict(zip(obj_cls2id.values(), obj_cls2id.keys()))

with open('data_format/{:s}_rel.txt'.format(args.imds), 'r') as f:
    rel_clss = [rel.strip() for rel in f.readlines()]
rel_mat = np.zeros([len(obj_cls2id), len(rel_clss), len(obj_cls2id)], dtype=int)
obj_mat = np.zeros(len(obj_cls2id), dtype=int)
rel_cls2id = dict(zip(rel_clss, range(len(rel_clss))))


def statistic():
    annodir = args.imds + '/' + 'annotation/train'
    for i, annofile in enumerate(os.listdir(annodir)):
        if i % 500:
            print(i)
        anno = json.load(open(annodir + '/' + annofile))
        so_insts = anno['subject/objects']
        rel_insts = anno['relation_instances']

        so_dict = dict()  # {tid: clsid; ...}
        for so_inst in so_insts:
            tid = so_inst['tid']
            clsname = so_inst['category']
            clsid = obj_cls2id[clsname]
            so_dict[tid] = clsid
            obj_mat[clsid] += 1
        for rel_inst in rel_insts:
            s_tid = rel_inst['subject_tid']
            o_tid = rel_inst['object_tid']
            rel_name = rel_inst['predicate']

            rel_id = rel_cls2id[rel_name]
            scls_id = so_dict[s_tid]
            ocls_id = so_dict[o_tid]
            rel_mat[scls_id, rel_id, ocls_id] += 1

    data = {'rel': rel_mat, 'obj': obj_mat}
    return data


def data_analysis(data):
    rel_mat = data['rel']
    obj_mat = data['obj']
    print('Instance number of each obj category')
    for i in range(obj_num):
        print('{:s}: {:d}'.format(obj_id2cls[i], obj_mat[i]))

    print('Instance number of each relation')
    for i in range(rel_num):
        print('{:s}: {:d}'.format(rel_cls2id[i], np.sum(rel_mat, axis=[0, 2])))


def test_vid_len():
    bar = tqdm(os.listdir(args.imds + '/test/video'))
    ave = 0.
    vnum = 0
    for subdir in bar:
        bar.set_description('Test average len')
        subpath = args.imds + '/video/' + subdir
        for vid in os.listdir(subpath):
            videofile = subpath + '/' + vid
            vcap = cv2.VideoCapture(videofile)
            if not vcap.isOpened():
                print("cannot open %s" % videofile)
            else:
                fnum = vcap.get(7)
                vnum += 1
                ave += fnum
    print(vnum)
    print(ave/vnum)


def cal_proposal_num(dbname='vidvrd'):
    if dbname == 'vidvrd':
        vidvrd_txt = [#'vidvrd/ImageSets/Main/train.txt',
                      #'coco/ImageSets/Main/train_vidvrd.txt',
                      'imagenet_det/ILSVRC2015/ImageSets/DET/train_vidvrd.txt'
                     ]
        vidvrd_txt = vidvrd_txt[0]
        with open(vidvrd_txt, 'r') as f:
            files = [file.strip() for file in f.readlines()]
        cls_stat = dict()
        cls_list = ["turtle", "antelope", "bicycle", "lion", "ball", "motorcycle", "cattle", 
        "airplane", "red_panda", "horse", "watercraft", "monkey", "fox", "elephant", "bird", 
        "sheep", "frisbee", "giant_panda", "squirrel", "bus", "bear", "tiger", "train", "snake",
        "rabbit", "whale", "sofa", "skateboard", "dog", "domestic_cat", "person",
        "lizard", "hamster", "car", "zebra"]
        clscode2clsname = {
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
        for c in cls_list:
            cls_stat[c] = 0
        dataset = vidvrd_txt.split('/')[0]
        for file in tqdm(files):
            if dataset == 'vidvrd':
                vid = file.split('-')[0]
                fid = file.split('-')[1]
                xmlfile = os.path.join('vidvrd/Annotations/train', vid, str(int(fid))+'.xml')
            elif dataset == 'coco':
                xmlfile = os.path.join('coco/Annotations/train_vidvrd', file.strip()+'.xml')
            elif dataset == 'imagenet_det':
                #year = int(file.split('/')[0].split('_')[0][6:])
                xmlfile = os.path.join('imagenet_det/ILSVRC2015/Annotations/DET/train_vidvrd', file+'.xml')
            else:
                raise NotImplementedError
            tree = ET.parse(xmlfile)
            objs = tree.findall('object')
            for obj in objs:
                name = obj.find('name').text
                if dataset == 'imagenet_det':
                    if clscode2clsname[name] not in cls_stat:
                        raise IOError
                    cls_stat[clscode2clsname[name]] += 1
                else:
                    cls_stat[name] += 1
    else:
        # calculate each category instance num
        # by the way I check if the image H and W match the annotation for ImageNet DET
        vidor_txt = ['vidor/ImageSets/Main/train.txt',
                    #'coco/ImageSets/Main/train_vidor.txt',
                    #'imagenet_det/ILSVRC2015/ImageSets/DET/train_vidor.txt'
                    ]
        vidor_txt = vidor_txt[0]
        with open(vidor_txt, 'r') as f:
            files = [file.strip() for file in f.readlines()]
        
        cls_stat = dict()
        cls_list = ['bread', 'cake', 'dish', 'fruits', 'vegetables', 'backpack', 'camera',
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
                    'squirrel', 'tiger', 'adult', 'baby', 'child']
        clscode2clsname = {
                'n07745940': 'fruits', 'n07768694': 'fruits', 'n07753275': 'fruits', 'n07747607': 'fruits',
                'n07749582': 'fruits', 'n07753592': 'fruits', 'n07739125': 'fruits', 'n07753113': 'fruits', 'n07714571': 'vegetables',
                'n02769748': 'backpack', 'n03642806': 'laptop', 'n02802426': 'ball/sports_ball', 'n03134739': 'ball/sports_ball',
                'n03445777': 'ball/sports_ball', 'n03942813': 'ball/sports_ball', 'n04118538': 'ball/sports_ball',
                'n02799071': 'ball/sports_ball', 'n04254680': 'ball/sports_ball', 'n04409515': 'ball/sports_ball',
                'n04540053': 'ball/sports_ball', 'n04039381': 'racket', 'n04228054': 'ski', 'n03001627': 'chair',
                'n03797390': 'cup', 'n03271574': 'electric_fan', 'n03761084': 'microwave', 'n04070727': 'refrigerator',
                'n03211117': 'screen/monitor', 'n04256520': 'sofa', 'n04379243': 'table', 'n03467517': 'guitar',
                'n03928116': 'piano', 'n02828884': 'bench', 'n06874185': 'traffic_light', 'n02691156': 'aircraft',
                'n02834778': 'bicycle', 'n02924116': 'bus/truck', 'n02958343': 'car', 'n03790512': 'motorcycle',
                'n04468005': 'train', 'n04530566': 'watercraft', 'n01503061': 'bird', 'n01726692': 'snake',
                'n01662784': 'turtle', 'n02419796': 'antelope', 'n02131653': 'bear', 'n02437136': 'camel',
                'n02121808': 'cat', 'n02402425': 'cattle/cow', 'n02084071': 'dog', 'n02503517': 'elephant',
                'n02342885': 'hamster/rat', 'n02374451': 'horse', 'n02129165': 'lion', 'n02510455': 'panda',
                'n02324045': 'rabbit', 'n02411705': 'sheep/goat', 'n02355227': 'squirrel', 'n02129604': 'tiger',
                'n04591713': 'bottle', 'n04557648': 'bottle'
            }
        for c in cls_list:
            cls_stat[c] = 0

        for file in tqdm(files):
            if vidor_txt.split('/')[0] == 'vidor':
                vid = file.split('-')[0]
                sub = file.split('-')[1]
                fid = file.split('-')[2]
                xmlfile = os.path.join('vidor/Annotations/train', sub, vid, str(int(fid))+'.xml')
            elif vidor_txt.split('/')[0] == 'coco':
                xmlfile = os.path.join('coco/Annotations/train_vidor', file.strip()+'.xml')
                jpgfile = os.path.join('coco/train2017', file.strip()+'.jpg')
            elif vidor_txt.split('/')[0] == 'imagenet_det':
                #year = int(file.split('/')[0].split('_')[0][6:])
                xmlfile = os.path.join('imagenet_det/ILSVRC2015/Annotations/DET/train_vidor', file+'.xml')
                jpgfile = os.path.join('imagenet_det/ILSVRC2015/Data/DET/train', file+'.JPEG')
            else:
                raise NotImplementedError
            tree = ET.parse(xmlfile)
            objs = tree.findall('object')
            for obj in objs:
                name = obj.find('name').text
                #if clscode2clsname[name] not in cls_stat:
                #    raise IOError
                #cls_stat[clscode2clsname[name]] += 1
                cls_stat[name] += 1
            if vidor_txt.split('/')[0] != 'vidor':
                im = cv2.imread(jpgfile)
                h, w, _ = im.shape
                size = tree.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                if h != height or w != width:
                    print(file)

    for c in cls_stat:
        print(c)
    for c in cls_stat:
        print(cls_stat[c])


# ---------------------------------------
# read videos and write them into frames
# ---------------------------------------
def write_frames(vcap, videoname, video_out_path, frame_gap):
    frame_count = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
    cur_frame = 0
    flist = []
    while cur_frame < frame_count:
        suc, frame = vcap.read()
        if not suc:
            cur_frame += 1
            print("warning, %s frame of %s failed" % (cur_frame, videoname))
            continue
        if cur_frame % frame_gap != 0:
            cur_frame += 1
            continue
        im = frame.astype("float32")
        frame_file = os.path.join(video_out_path, "%05d.jpg" % cur_frame)
        flist.append(frame_file)
        cv2.imwrite(frame_file, im)
        cur_frame += 1
    return flist


import glob
def get_frames():
    # get test video ids
    if args.imds == 'vidvrd':
        testdir = args.imds + '/annotation/test'
        testvids = [vid.split('.')[0] for vid in os.listdir(testdir)]
        for vid in tqdm(os.listdir(args.imds + '/video')):
            frame_gap = 1
            if vid.split('.')[0] not in testvids:
                frame_gap = args.frame_gap
    
            try:
                videofile = args.imds + '/video/' + vid
                vcap = cv2.VideoCapture(videofile)
                if not vcap.isOpened():
                    raise Exception("cannot open %s" % videofile)
            except Exception as e:
                raise e

            videoname = os.path.basename(videofile).split('.')[0]
            video_out_path = os.path.join('vidvrd/JPEGImages', videoname)
            if not os.path.exists(video_out_path):
                os.makedirs(video_out_path)
            write_frames(vcap, videoname, video_out_path, frame_gap)
    elif args.imds == 'vidor':
        test = True
        testlist = []
            
        if test:
            frame_gap = 1
            bar = tqdm(os.listdir(args.imds + '/test/video'))
            for subdir in bar:
                bar.set_description('Test')
                subpath = args.imds + '/test/video/' + subdir
                for vid in os.listdir(subpath):
                    imgsdir = 'vidor/test/JPEGImages/%s/%s/'%(subdir, vid[:-4])
                    if os.path.exists(imgsdir):
                        #imgs = glob.glob(imgsdir+'*.jpg')
                        imgs = sorted([im.strip()[:-4] for im in os.listdir(imgsdir)])
                        flist = [imgsdir+im+'.jpg' for im in imgs]
                        testlist += flist
                        continue
                    #else:
                    #    print(subdir, vid)
                    #    assert 1==0
                    videofile = subpath + '/' + vid
                    vcap = cv2.VideoCapture(videofile)
                    if not vcap.isOpened():
                        print("cannot open %s" % videofile)
                    videoname = os.path.basename(videofile).split('.')[0]
                    video_out_path = os.path.join('vidor/test/JPEGImages/' + subdir + '/', videoname)
                    if not os.path.exists(video_out_path):
                        os.makedirs(video_out_path)
                    flist = write_frames(vcap, videoname, video_out_path, frame_gap)
                    testlist += flist
            with open('vidor/test/test.txt', 'w') as f:
                for t in testlist:
                    f.writelines(t+'\n')
        else:
            valdir = args.imds + '/annotation/val/'
            valvids = []
            for subdir in os.listdir(valdir):
                valvids += [vid.split('.')[0] for vid in os.listdir(valdir+subdir)]
            badfiles = []
            for subdir in tqdm(os.listdir(args.imds + '/video')):
                subpath = args.imds + '/video/' + subdir
                for vid in os.listdir(subpath):
                    frame_gap = 1
                    if vid.split('.')[0] not in valvids:
                        continue
                        frame_gap = args.frame_gap
                        
                    videofile = subpath + '/' + vid
                    vcap = cv2.VideoCapture(videofile)
                    if not vcap.isOpened():
                        badfiles.append(vid)
                        print("cannot open %s" % videofile)
                        continue

                    videoname = os.path.basename(videofile).split('.')[0]
                    video_out_path = os.path.join('vidor/JPEGImages/' + subdir + '/', videoname)
                    if not os.path.exists(video_out_path):
                        os.makedirs(video_out_path)

                    write_frames(vcap, videoname, video_out_path, frame_gap)

            with open('{:s}_badfiles.txt'.format(args.imds), 'w') as f:
                for badfile in badfiles:
                    f.writelines(badfile+'/n')


# -------------------------------
# generate xml annotations
# -------------------------------
def make_dom(subdir, im_id, ih, iw, boxes, tids, clses, generated):
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = str(subdir)
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = str(im_id)
    node_source = SubElement(node_root, 'source')
    node_dataset = SubElement(node_source, 'dataset')
    node_dataset.text = args.imds

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(iw)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(ih)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for i, box in enumerate(boxes):
        node_object = SubElement(node_root, 'object')
        node_trackid = SubElement(node_object, 'trackid')
        node_trackid.text = str(tids[i])
        node_name = SubElement(node_object, 'name')
        node_name.text = clses[i]

        xmin, ymin, xmax, ymax = box
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(xmin)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(ymin)
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(xmax)
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(ymax)

        node_occluded = SubElement(node_object, 'occluded')
        node_occluded.text = '0'
        node_generated = SubElement(node_object, 'generated')
        node_generated.text = str(generated[i])

    xml = tostring(node_root)
    dom = parseString(xml)
    return dom


def write_xmls(subdir, jsondir, xml_dir, json_name):
    json_path = jsondir + '/' + json_name
    with open(json_path, 'r') as f:
        data = json.load(f)

    ih = data['height']
    iw = data['width']
    objs = data['subject/objects']
    for num, frame in enumerate(data['trajectories']):
        #print(frame)
        clss = []
        boxes = []
        tids = []
        generated = []
        for box in frame:
            # print(box)
            for obj in objs:
                if box['tid'] == obj['tid']:
                    clss.append(obj['category'])
            xmin = box['bbox']['xmin']
            ymin = box['bbox']['ymin']
            xmax = box['bbox']['xmax']
            ymax = box['bbox']['ymax']
            boxes.append([xmin, ymin, xmax, ymax])
            tids.append(box['tid'])
            generated.append(frame[0]['generated'])
        dom = make_dom(subdir, num, ih, iw, boxes, tids, clss, generated)
        xml_name = os.path.join(xml_dir, str(num) + '.xml')
        with open(xml_name, 'wb') as f:
            f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))


def get_xmls():
    for mode in modes:
        json_num = 0
        root_dir = args.imds+'/annotation/'+mode
        if args.imds == 'vidor':
            for subdir in os.listdir(root_dir):
                subpath = root_dir + subdir
                xml_subdir = 'vidor/Annotations/'+mode+'/'+ subdir
                if not os.path.exists(xml_subdir):
                    os.makedirs(xml_subdir)
                for json_name in os.listdir(subpath):
                    xml_dir = xml_subdir+'/'+json_name[:-5]
                    if not os.path.exists(xml_dir):
                        os.makedirs(xml_dir)
                    json_num += 1
                    if json_num % 100 == 0:
                        print(json_num)
                    write_xmls(subdir, subpath, xml_dir, json_name)
        elif args.imds == 'vidvrd':
            for json_name in os.listdir(root_dir):
                print(json_name)
                xml_dir = 'vidvrd/Annotations/'+mode+'/'+json_name[:-5]
                if not os.path.exists(xml_dir):
                    os.makedirs(xml_dir)
                json_num += 1
                if json_num % 100 == 0:
                    print(json_num)
                write_xmls('None', root_dir, xml_dir, json_name)
        else:
            raise NotImplementedError

                
# ----------------------------------------------
# spilit dataset into trainval.txt and test.txt
# ----------------------------------------------
def get_spilit():
    print('Get split', args.imds)
    if args.imds == 'vidvrd':
        modesets = ['train', 'test']  # val.txt is 1/8 of test.txt
    else:
        modesets = ['train']
    for mode in modesets:
        rootdir = args.imds+'/ImageSets/Main/'
        if not os.path.exists(rootdir):
            os.makedirs(rootdir)
        f = open(rootdir+mode+'.txt', 'w')
        frame_list = list()
        if args.imds == 'vidvrd':
            for vid in tqdm(os.listdir('vidvrd/annotation/{:s}'.format(mode))):
                for frame in os.listdir('vidvrd/JPEGImages/'+vid.split('.')[0]):
                    frame_id = vid.split('.')[0]+'-'+frame.split('.')[0]
                    frame_list.append(frame_id)
        elif args.imds == 'vidor':
            for subdir in tqdm(os.listdir('vidor/annotation/{:s}'.format(mode))):
                for vid in os.listdir('vidor/annotation/{:s}/'.format(mode)+subdir):
                    for frame in os.listdir('vidor/JPEGImages/'+subdir+'/'+vid.split('.')[0]):
                        if mode == 'train' and int(frame.split('.')[0]) % 32 != 0:
                            continue
                        xml_file = os.path.join('vidor/Annotations/{:s}/{:s}/{:s}/{:d}.xml'.
                                                format(mode, subdir, vid.split('.')[0], int(frame.split('.')[0])))
                        frame_id = vid.split('.')[0] + '-' + subdir + '-' + frame.split('.')[0]
                        try:
                            tree = ET.parse(xml_file)
                        except:
                            print(xml_file)
                        objs = tree.findall('object')
                        if objs == 0:
                            continue
                        frame_list.append(frame_id)
        else:
            raise NotImplementedError

        random.shuffle(frame_list)
        for frame_id in frame_list:
            f.writelines(frame_id+'\n')
        f.close()
    
    print('filter out some empty image samples (no bbox)')
    if args.imds == 'vidvrd':
        for mode in ['test']:
            new_imagelist = []
            f = open('vidvrd/ImageSets/Main/{:s}.txt'.format(mode), 'r')
            imagelist = [image.strip() for image in f.readlines()]
            for image in tqdm(imagelist):
                vid_name = image.split('-')[0]
                frame_id = int(image.split('-')[1])
                annofile = 'vidvrd/Annotations/{:s}/{:s}/{:d}.xml'.format(mode, vid_name, frame_id)
                if not os.path.exists(annofile):
                    continue

                tree = ET.parse(annofile)
                objs = tree.findall('object')

                new_imagelist.append(image)
            f.close()
            with open('vidvrd/ImageSets/Main/{:s}.txt'.format(mode), 'w') as f:
                for image in new_imagelist:
                    f.writelines(image+'\n')
    elif args.imds == 'vidor':
        for mode in modes:
            new_imagelist = []
            f = open('vidor/ImageSets/Main/{:s}.txt'.format(mode), 'r')
            imagelist = [image.strip() for image in f.readlines()]
            for image in tqdm(imagelist):
                vid_name = image.split('-')[0]
                sub_dir = image.split('-')[1]
                frame_id = int(image.split('-')[2])
                annofile = 'vidor/Annotations/{:s}/{:s}/{:s}/{:d}.xml'.format(mode, sub_dir, vid_name, frame_id)
                if not os.path.exists(annofile):
                    continue
                tree = ET.parse(annofile)
                objs = tree.findall('object')
                if len(objs) == 0:
                    continue
                new_imagelist.append(image)
            f.close()
            with open('vidor/ImageSets/Main/{:s}.txt'.format(mode), 'w') as f:
                for image in new_imagelist:
                    f.writelines(image+'\n')
    #'''

    
# ------------------------------------------------------------
# regenerate ImageSets train_vidvrd.txt and vidor.txt for ImageNet-DET dataset
# some missing and mistakes exist in raw annotations
# Also, we only select object categories for vidor and vidvrd
# Make sure the raw annotation dirs exist under rootdir
#
#  vidvrd object category list
#  "turtle", "antelope", "bicycle", "lion", "ball", "motorcycle", "cattle", "airplane", "red_panda",
#  "horse", "watercraft", "monkey", "fox", "elephant", "bird", "sheep", "frisbee", "giant_panda",
#  "squirrel", "bus", "bear", "tiger", "train", "snake", "rabbit", "whale", "sofa", "skateboard", "dog",
#  "domestic_cat", "person", "lizard", "hamster", "car", "zebra")  35
# ------------------------------------------------------------
def re_imageset():
    imageset_list = []
    if args.imds == 'vidvrd':
        rootdir = 'imagenet_det/ILSVRC2015/Annotations/DET/train'
        cls_indexs = ['n01662784', 'n02419796', 'n02834778', 'n02129165', 'n02691156', 'n02802426', 'n03134739',
                        'n03445777', 'n03942813', 'n04118538', 'n04254680', 'n04409515', 'n04540053', 'n03790512',
                        'n02402425', 'n02509815', 'n02374451', 'n04530566', 'n02484322', 'n02118333', 'n02503517',
                        'n01503061', 'n02411705', 'n02510455', 'n02355227', 'n02924116', 'n02131653', 'n02129604',
                        'n04468005', 'n01726692', 'n02324045', 'n02062744', 'n04256520', 'n02084071', 'n02121808',
                        'n00007846', 'n01674464', 'n02342885', 'n02958343', 'n02391049', 'n02799071']
        filenum = 0
        for subdir in tqdm(os.listdir(rootdir)):
            if subdir == 'ILSVRC2013_train': # imagenet_det_2013
                for cls in os.listdir(rootdir+'/'+subdir):
                    if cls not in cls_indexs:
                        continue
                    for xmlfile in os.listdir(rootdir+'/'+subdir+'/'+cls):
                        imgid = xmlfile.split('.')[0]
                        imageset_list.append(subdir+'/'+cls+'/'+imgid)
                        filenum += 1
            else:
                # we filter out those categoties which don't belong to VIDVRD
                for filename in os.listdir(rootdir+'/'+subdir):
                    filepath = rootdir+'/'+subdir+'/'+filename
                    tree = ET.parse(filepath)
                    objs = tree.findall('object')
                    obj_ids = [obj.find('name').text for obj in objs]
                    keep = False
                    for obj_id in obj_ids:
                        if obj_id in cls_indexs:
                            keep = True
                    if keep:
                        imageset_list.append(subdir+'/'+filename.split('.')[0])
                        filenum += 1
        with open('imagenet_det/ILSVRC2015/ImageSets/DET/train_vidvrd.txt', 'w') as f:
            for image in imageset_list:
                f.writelines(image+'\n')
        print(filenum)
    elif args.imds == 'vidor':
        rootdir = 'imagenet_det/ILSVRC2015/Annotations/DET/train'
        cls_indexs = ['n07745940', 'n07768694', 'n07753275', 'n07747607', 'n07749582', 'n07753592', 'n07739125',
                      'n07753113', 'n07714571', 'n02769748', 'n03642806', 'n02802426', 'n03134739', 'n03445777',
                      'n03942813', 'n04118538', 'n02799071', 'n04254680', 'n04409515', 'n04540053', 'n04039381',
                      'n04228054', 'n03001627', 'n03797390', 'n03271574', 'n03761084', 'n04070727', 'n03211117',
                      'n04256520', 'n04379243', 'n03467517', 'n03928116', 'n02828884', 'n06874185', 'n02691156',
                      'n02834778', 'n02924116', 'n02958343', 'n03790512', 'n04468005', 'n04530566', 'n01503061',
                      'n01726692', 'n01662784', 'n02419796', 'n02131653', 'n02437136', 'n02121808', 'n02402425',
                      'n02084071', 'n02503517', 'n02342885', 'n02374451', 'n02129165', 'n02510455', 'n02324045',
                      'n02411705', 'n02355227', 'n02129604']
        filenum = 0
        for subdir in tqdm(os.listdir(rootdir)):
            if subdir == 'ILSVRC2013_train': # imagenet_det_2013
                for cls in os.listdir(rootdir+'/'+subdir):
                    if cls not in cls_indexs:
                        continue
                    for xmlfile in os.listdir(rootdir+'/'+subdir+'/'+cls):
                        imgid = xmlfile.split('.')[0]
                        imageset_list.append(subdir+'/'+cls+'/'+imgid)
                        filenum += 1
            else:
                # we filter out those categoties which don't belong to VIDVRD
                for filename in os.listdir(rootdir+'/'+subdir):
                    filepath = rootdir+'/'+subdir+'/'+filename
                    tree = ET.parse(filepath)
                    objs = tree.findall('object')
                    obj_ids = [obj.find('name').text for obj in objs]
                    keep = False
                    for obj_id in obj_ids:
                        if obj_id in cls_indexs:
                            keep = True
                    if keep:
                        imageset_list.append(subdir+'/'+filename.split('.')[0])
                        filenum += 1
        with open('imagenet_det/ILSVRC2015/ImageSets/DET/train_vidor.txt', 'w') as f:
            for image in imageset_list:
                f.writelines(image+'\n')
        print(filenum)
    else:
        raise NotImplementedError

    # -------------------------------------------------
    # check whether the image is valid, some is empty
    # -------------------------------------------------
    with open('imagenet_det/ILSVRC2015/ImageSets/DET/train_vidvrd.txt', 'r') as f:
        imageset_list = [i.strip() for i in f.readlines()]
    print(len(imageset_list))
    for img in tqdm(imageset_list):
        year_set = img.split('/')[0].split('_')[0]
        if '2013' in year_set:
            jpgpath = 'imagenet_det/ILSVRC2015/Data/DET/train/{:s}/{:s}/{:s}.JPEG' \
                .format(img.split('/')[0], img.split('/')[1], img.split('/')[2])
        else:
            jpgpath = 'imagenet_det/ILSVRC2015/Data/DET/train/{:s}/{:s}.JPEG' \
                .format(img.split('/')[0], img.split('/')[1])

        data = cv2.imread(jpgpath)
        h,w,c = data.shape
        if h <= 0 or w <= 0 or c <= 0:
            print(jpgpath)
        if c != 3:
            print(jpgpath)
    
    


# -------------------------------------------------------
# filter out useless clss in xml file for ImageNet-DET
# -------------------------------------------------------
def filter_useless_cls():
    # copy category dir in ILSVRC2013 to train_vidvrd
    print('Dataset: ', args.imds)
    if args.imds == 'vivrd':
        cls_indexs = ['n01662784', 'n02419796', 'n02834778', 'n02129165', 'n02691156', 'n02802426', 'n03134739',
                      'n03445777', 'n03942813', 'n04118538', 'n04254680', 'n04409515', 'n04540053', 'n03790512',
                      'n02402425', 'n02509815', 'n02374451', 'n04530566', 'n02484322', 'n02118333', 'n02503517',
                      'n01503061', 'n02411705', 'n02510455', 'n02355227', 'n02924116', 'n02131653', 'n02129604',
                      'n04468005', 'n01726692', 'n02324045', 'n02062744', 'n04256520', 'n02084071', 'n02121808',
                      'n00007846', 'n01674464', 'n02342885', 'n02958343', 'n02391049', 'n02799071']
    elif args.imds == 'vidor':
        cls_indexs = ['n07745940', 'n07768694', 'n07753275', 'n07747607', 'n07749582', 'n07753592', 'n07739125',
                      'n07753113', 'n07714571', 'n02769748', 'n03642806', 'n02802426', 'n03134739', 'n03445777',
                      'n03942813', 'n04118538', 'n02799071', 'n04254680', 'n04409515', 'n04540053', 'n04039381',
                      'n04228054', 'n03001627', 'n03797390', 'n03271574', 'n03761084', 'n04070727', 'n03211117',
                      'n04256520', 'n04379243', 'n03467517', 'n03928116', 'n02828884', 'n06874185', 'n02691156',
                      'n02834778', 'n02924116', 'n02958343', 'n03790512', 'n04468005', 'n04530566', 'n01503061',
                      'n01726692', 'n01662784', 'n02419796', 'n02131653', 'n02437136', 'n02121808', 'n02402425',
                      'n02084071', 'n02503517', 'n02342885', 'n02374451', 'n02129165', 'n02510455', 'n02324045',
                      'n02411705', 'n02355227', 'n02129604']
    else:
        raise NotImplementedError

    print('ImageSet for 2013')
    old_dir = 'imagenet_det/ILSVRC2015/Annotations/DET/train/ILSVRC2013_train'
    new_dir = 'imagenet_det/ILSVRC2015/Annotations/DET/train_{:s}/ILSVRC2013_train'.format(args.imds)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for subdir in tqdm(os.listdir(old_dir)):
        if subdir in cls_indexs and not os.path.exists(new_dir + '/' + subdir):
            os.system('cp -r {:s}/{:s} {:s}'.format(old_dir, subdir, new_dir))

    print('ImageSet for 2014')
    imageset_file = 'imagenet_det/ILSVRC2015/ImageSets/DET/train_{:s}.txt'.format(args.imds)
    with open(imageset_file, 'r') as f:
        imageset_list = []
        for line in tqdm(f.readlines()):
            new_line = line.strip()  # ILSVRC2014_train_0006/ILSVRC2014_train_00060229
            image_set = new_line.split('/')[0].split('_')[0]
            #if '2013' in image_set:
            #    print(new_line)
            #    continue
            imageset_list.append(new_line)

    print('filter out xml cls')
    for image in tqdm(imageset_list):
        tree = ET.parse('imagenet_det/ILSVRC2015/Annotations/DET/train/' + image + '.xml')
        objs = tree.findall('object')
        root = tree.getroot()
        if len(objs) == 0:
            continue
        for obj in objs:
            if obj.find('name').text not in cls_indexs:
                root.remove(obj)

        new_dir = 'imagenet_det/ILSVRC2015/Annotations/DET/train_{:s}/'.format(args.imds) + image.split('/')[0]
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        tree.write('imagenet_det/ILSVRC2015/Annotations/DET/train_{:s}/'.format(args.imds) + image + '.xml',
                   encoding='utf-8', xml_declaration=True)


# ---------------------------------------------
# Transform the COCO dataset into VOC format
# ---------------------------------------------
coco_id_name_map={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                   6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                   11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                   16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                   22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                   28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                   35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                   40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                   44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                   51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                   56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                   61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                   70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                   77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                   82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                   88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


def format_coco():
    print('format COCO dataset into VOC')
    with open("coco/annotations/instances_train2017.json", 'r') as f:
        annos = json.load(f)['annotations']
    if args.imds == 'vidvrd':
        coco2vrd_clss = {'bicycle': 'bicycle', 'motorbike': 'motorcycle', 'cow': 'cattle', 'sports ball': 'ball',
                         'aeroplane': 'airplane', 'horse': 'horse', 'boat': 'watercraft', 'elephant': 'elephant',
                         'bird': 'bird', 'sheep': 'sheep', 'frisbee': 'frisbee', 'bus': 'bus', 'train': 'train',
                         'sofa': 'sofa', 'skateboard': 'skateboard', 'dog': 'dog', 'cat': 'domestic_cat',
                         'person': 'person', 'car': 'car', 'zebra': 'zebra'}
    elif args.imds == 'vidor':
        coco2vrd_clss = {'cake': 'cake', 'banana': 'fruits', 'apple': 'fruits', 'orange': 'fruits', 'backpack': 'backpack',
                         'laptop': 'laptop', 'suitcase': 'suitcase', 'sports ball': 'ball/sports_ball',
                         'baseball bat': 'bat', 'frisbee': 'frisbee', 'skateboard': 'skateboard', 'skis': 'ski',
                         'snowboard': 'snowboard', 'surfboard': 'surfboard', 'tennis racket': 'racket',
                         'cell phone': 'cellphone', 'bottle': 'bottle', 'chair':'chair', 'cup': 'cup', 'microwave': 'microwave',
                         'oven': 'oven', 'refrigerator': 'refrigerator', 'tvmonitor': 'screen/monitor', 'sink': 'sink',
                         'sofa': 'sofa', 'toilet': 'toilet', 'bench': 'bench', 'stop sign': 'stop_sign',
                         'traffic light':'traffic_light', 'aeroplane': 'aircraft', 'bicycle': 'bicycle', 'bus': 'bus/truck',
                         'car': 'car', 'motorbike': 'motorcycle', 'train': 'train', 'boat': 'watercraft', 'bird': 'bird',
                         'diningtable': 'table', 'bear': 'bear', 'cat': 'cat', 'cow': 'cattle/cow', 'dog': 'dog',
                         'elephant': 'elephant', 'horse': 'horse', 'sheep': 'sheep/goat',
                        }
    else:
        raise NotImplementedError

    imglist = []
    print('Generate imglist_{:s}.json'.format(args.imds))
    for anno in tqdm(annos):
        if coco_id_name_map[anno['category_id']] in coco2vrd_clss.keys():
            inst = {
                "filename": str(anno["image_id"]).zfill(6),
                "name": coco2vrd_clss[coco_id_name_map[anno["category_id"]]],
                "bndbox": anno["bbox"]
            }
            imglist.append(inst)
    with open('coco/annotations/imglist_{:s}.json'.format(args.imds), 'w') as f:
        json.dump(imglist, f)

    print('Group object proposal by img_id and generate imageset list')
    with open('coco/annotations/imglist_{:s}.json'.format(args.imds), 'r') as f:
        imglist = json.load(f)
    img_dict = {}
    imageset_list = []
    for img in tqdm(imglist):
        # {'filename': '495357', 'name': 'dog', 'bndbox': [337.02, 244.46, 66.47, 66.75]}
        img_id = img['filename']
        imageset_list.append(img_id.zfill(12))
        if img_id not in img_dict.keys():
            img_dict[img_id] = []
        img_dict[img_id].append(img)
    imageset_list = list(set(imageset_list))
    print('Image Num: {:d}'.format(len(imageset_list)))
    with open('coco/ImageSets/Main/train_{:s}.txt'.format(args.imds), 'w') as f:
        for img in imageset_list:
            f.writelines(img + '\n')

    print('Generate xml-format annotations for COCO')
    for img_id in tqdm(img_dict.keys()):
        raw_boxes = img_dict[img_id]
        boxes = []
        clss = []
        for box in raw_boxes:
            boxes.append([box['bndbox'][0], box['bndbox'][1],
                          box['bndbox'][0] + box['bndbox'][2], box['bndbox'][1] + box['bndbox'][3]])
            clss.append(box['name'])
        tids = [-1 for i in range(len(raw_boxes))]
        generated = [0 for i in range(len(raw_boxes))]
        img = cv2.imread('coco/train2017/' + img_id.zfill(12) + '.jpg')
        ih, iw, _ = img.shape
        dom = make_dom('train2017', img_id, ih, iw, boxes, tids, clss, generated)
        xml_name = os.path.join('coco/Annotations/train_{:s}/'.format(args.imds), img_id.zfill(12) + '.xml')
        with open(xml_name, 'wb') as f:
            f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))


# --------------------------------------------------------------------
# we check 10 image annos for vidvrd, vidor, imagenet_det and coco dataset
# --------------------------------------------------------------------
def vis_detections(xmlpath, jpgpath):
    """Visual debugging of detections."""
    tree = ET.parse(xmlpath)
    objs = tree.findall('object')
    boxes, clss = [], []
    for obj in objs:
        clss.append(obj.find('name').text)
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        boxes.append([x1, y1, x2, y2])
    im = cv2.imread(jpgpath)
    for i in range(np.minimum(10, len(boxes))):
        bbox = tuple(int(np.round(x)) for x in boxes[i])
        cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
        cv2.putText(im, '%s' % clss[i], (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im


def check_anno():
    # check vidvrd and vidor first
    for dataset in ['vidvrd', 'vidor']:  # vidvrd and vidor
        if not os.path.exists('imgs/{:s}'.format(dataset)):
            os.makedirs('imgs/{:s}'.format(dataset))
        if dataset == 'vidvrd':
            modes = ['train', 'test']
        else:
            modes = ['train', 'val']
        for mode in modes:
            imageset_file = '{:s}/ImageSets/Main/{:s}.txt'.format(dataset, mode)
            with open(imageset_file, 'r') as f:
                imgs = [file.strip() for file in f.readlines()][100:110]
                for img in imgs:
                    if dataset == 'vidvrd':
                        xmlpath = 'vidvrd/Annotations/{:s}/{:s}/{:d}.xml'.format(mode, img.split('-')[0],
                                                                                 int(img.split('-')[1]))
                        jpgpath = 'vidvrd/JPEGImages/{:s}/{:s}.jpg'.format(img.split('-')[0], img.split('-')[1])
                    elif dataset == 'vidor':
                        xmlpath = 'vidor/Annotations/{:s}/{:s}/{:s}/{:d}.xml'.format(mode, img.split('-')[1],
                                                                                     img.split('-')[0],
                                                                                     int(img.split('-')[2]))
                        jpgpath = 'vidor/JPEGImages/{:s}/{:s}/{:s}.jpg'.format(img.split('-')[1], img.split('-')[0],
                                                                               img.split('-')[2])

                    show_img = vis_detections(xmlpath, jpgpath)
                    savepath = 'imgs/{:s}/{:s}'.format(dataset, mode)
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)
                    #print(show_img.shape)
                    cv2.imwrite(savepath+'/'+img+'.jpg', show_img)

    # then check imagenet_det and coco
    for ext_set in ['imagenet_det', 'coco']:
        for target_set in ['vidvrd', 'vidor']:
            if ext_set == 'imagenet_det':
                imageset_file = 'imagenet_det/ILSVRC2015/ImageSets/DET/train_{:s}.txt'.format(target_set)
            else:
                imageset_file = 'coco/ImageSets/Main/train_{:s}.txt'.format(target_set)

            with open(imageset_file, 'r') as f:
                imgs = [file.strip() for file in f.readlines()][100:110]
                for img in imgs:
                    if ext_set == 'imagenet_det':
                        year_set = img.split('/')[0].split('_')[0]
                        if '2013' in year_set:
                            xmlpath = 'imagenet_det/ILSVRC2015/Annotations/DET/train_{:s}/{:s}/{:s}/{:s}.xml' \
                                .format(target_set, img.split('/')[0], img.split('/')[1], img.split('/')[2])
                            jpgpath = 'imagenet_det/ILSVRC2015/Data/DET/train/{:s}/{:s}/{:s}.JPEG' \
                                .format(img.split('/')[0], img.split('/')[1], img.split('/')[2])
                        else:  # 2014
                            xmlpath = 'imagenet_det/ILSVRC2015/Annotations/DET/train_{:s}/{:s}/{:s}.xml' \
                                .format(target_set, img.split('/')[0], img.split('/')[1])
                            jpgpath = 'imagenet_det/ILSVRC2015/Data/DET/train/{:s}/{:s}.JPEG' \
                                .format(img.split('/')[0], img.split('/')[1])
                    elif ext_set == 'coco':
                        xmlpath = 'coco/Annotations/train_{:s}/{:s}.xml'.format(target_set, img.zfill(12))
                        jpgpath = 'coco/train2017/{:s}.jpg'.format(img.zfill(12))
                    else:
                        raise NotImplementedError

                    show_img = vis_detections(xmlpath, jpgpath)
                    savepath = 'imgs/{:s}/{:s}'.format(ext_set, target_set)
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)
                    cv2.imwrite(savepath + '/' + img.split('/')[-1] + '.jpg', show_img)


# -------------------------------------------------------------------------------
# We found some images and their xmlfile annos are wrong 
# (ImageNet DET and COCO)
# so delete them
# -------------------------------------------------------------------------------
def detele_imgs():
    # imageset
    dataset = args.imds
    print('Start ', dataset)
    wrongs = []
    del_db = 'imagenet_det'
    if del_db == 'imagenet_det':
        with open('imagenet_det/ILSVRC2015/ImageSets/DET/train_{:s}.txt'.format(dataset), 'r') as f:
            imageset_list = [i.strip() for i in f.readlines()]
        for img in tqdm(imageset_list):
            year_set = img.split('/')[0].split('_')[0]
            if '2013' in year_set:
                xmlpath = 'imagenet_det/ILSVRC2015/Annotations/DET/train_{:s}/{:s}/{:s}/{:s}.xml' \
                    .format(dataset, img.split('/')[0], img.split('/')[1], img.split('/')[2])
                jpgpath = 'imagenet_det/ILSVRC2015/Data/DET/train/{:s}/{:s}/{:s}.JPEG' \
                    .format(img.split('/')[0], img.split('/')[1], img.split('/')[2])
            else:  # 2014
                xmlpath = 'imagenet_det/ILSVRC2015/Annotations/DET/train_{:s}/{:s}/{:s}.xml' \
                    .format(dataset, img.split('/')[0], img.split('/')[1])
                jpgpath = 'imagenet_det/ILSVRC2015/Data/DET/train/{:s}/{:s}.JPEG' \
                    .format(img.split('/')[0], img.split('/')[1])
            imgdata = cv2.imread(jpgpath)
            im_h, im_w, _ = imgdata.shape
            imgdata = Image.open(jpgpath)
            w, h = imgdata.size
            try:
                tree = ET.parse(xmlpath)
            except:
                print('xml not found', img)
                imageset_list.remove(img)
                continue
            size = tree.find('size')
            xml_w = int(size.find('width').text)
            xml_h = int(size.find('height').text)
            #if im_h == im_w:
            #    print('H = W', img)
            if im_h != h or im_w != w:
                print(img)
                wrongs.append(img)
            if im_h != xml_h or im_w != xml_w:
                print(img)
                wrongs.append(img)
        for wrong in wrongs:
            imageset_list.remove(wrong)
        with open('imagenet_det/ILSVRC2015/ImageSets/DET/train_{:s}.txt'.format(dataset), 'w') as f:
            for img in imageset_list:
                f.writelines(img+'\n')
    elif del_db == 'coco':
        # delete coco
        datasets = ['vidor']
        squa = 0

        print('Start', dataset)
        with open('coco/ImageSets/Main/train_{:s}.txt'.format(dataset), 'r') as f:
            imageset_list = [i.strip() for i in f.readlines()]
        wrongs = []
        for img in tqdm(imageset_list):
            xmlpath = 'coco/Annotations/train_{:s}/{:s}.xml'.format(dataset, img.zfill(12))
            jpgpath = 'coco/train2017/{:s}.jpg'.format(img.zfill(12))
            tree = ET.parse(xmlpath)
            size = tree.find('size')
            xml_w = int(size.find('width').text)
            xml_h = int(size.find('height').text)
            imgdata = cv2.imread(jpgpath)
            im_h, im_w, _ = imgdata.shape
            if xml_w == xml_h:
                squa += 1
            if xml_w != im_w or xml_h != im_h:
                print(img)
                wrongs.append(img)
        for wrong in wrongs:
            imageset_list.remove(wrong)
        with open('coco/ImageSets/Main/train_{:s}.txt'.format(dataset), 'w') as f:
            for img in imageset_list:
                f.writelines(img+'\n')
        print('COCO H = W number: {:d}'.format(squa))
    else:
        # delete vidor (vidvrd is ok)
        with open('vidor/ImageSets/Main/train.txt', 'r') as f:
            imageset_list = [i.strip() for i in f.readlines()]
        squa = 0
        wrongs = []
        print('Start VIDOR')
        for img in tqdm(imageset_list):
            vid = img.split('-')[0]
            subdir = img.split('-')[1]
            frameid = img.split('-')[2]
            xmlpath = 'vidor/Annotations/train/{:s}/{:s}/{:d}.xml'.format(subdir, vid, int(frameid))
            jpgpath = 'vidor/JPEGImages/{:s}/{:s}/{:s}.jpg'.format(subdir, vid, frameid)
            tree = ET.parse(xmlpath)
            size = tree.find('size')
            xml_w = int(size.find('width').text)
            xml_h = int(size.find('height').text)
            #imgdata = cv2.imread(jpgpath)
            imagedata = Image.open(jpgpath)
            im_w, im_h = imagedata.size
            if xml_w == xml_h:
                squa += 1
            if xml_w != im_w or xml_h != im_h:
                print(img)
                wrongs.append(img)
        for wrong in wrongs:
            imageset_list.remove(wrong)
        with open('vidor/ImageSets/Main/train.txt', 'w') as f:
            for img in imageset_list:
                f.writelines(img + '\n')
        print('COCO H = W number: {:d}'.format(squa))


# ---------------------------------------------------------------
# Restrict the maxinum number of training samples for a category
# each category < 30000
# for vidvrd:
# person: we only select from COCO
# ---------------------------------------------------------------
def filter_toomany():
    # vidor
    vidor_txt = 'vidor/ImageSets/Main/train.txt'
    coco_txt = 'coco/ImageSets/Main/train_vidor.txt'   
    with open(vidor_txt, 'r') as f:
        files = [file.strip() for file in f.readlines()]
    print(len(files))
    dbname = 'vidor'
    if dbname == 'vidor':
        human_cls = ['adult', 'child', 'baby']
        stat, f_stat = 0, 0
        human_stat = {'adult':0, 'child':0, 'baby':0}
        new_files = []
        filter_clss = ['toy', 'bottle', 'chair', 'cup', 'screen/monitor', 'sofa', 'table', 'guitar', 
                        'cat', 'dog', 'adult', 'child', 'baby']
        filter_stat = dict()
        for c in filter_clss:
            filter_stat[c] = 0

        for file in tqdm(files):
            vid = file.split('-')[0]
            sub = file.split('-')[1]
            fid = file.split('-')[2]
            xmlfile = os.path.join('vidor/Annotations/train', sub, vid, str(int(fid))+'.xml')
            tree = ET.parse(xmlfile)
            objs = tree.findall('object')
            flag_h = 1
            flag_filter = 1
            for obj in objs:
                name = obj.find('name').text
                if name not in filter_clss or int(fid) % 128 == 0:
                    flag_filter = 0
            #if flag:
            #    # only contain human
            #    for obj in objs:
            #        name = obj.find('name').text
            #        human_stat[name] += 1
            #    stat += 1
            if not flag_filter:
                new_files.append(file)
        with open('vidor/ImageSets/Main/train_new.txt', 'w') as f:
            for file in new_files:
                f.writelines(file+'\n')
    elif dbname == 'coco':
        num = 0
        new_files = []
        cls_list = ['bread', 'cake', 'dish', 'fruits', 'vegetables', 'backpack', 'camera',
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
                    'squirrel', 'tiger', 'adult', 'baby', 'child']
        filter_clss = ['cake', 'fruits', 'backpack', 'laptop', 'suitcase', 'ball/sports_ball',
                        'baby_seat', 'bottle', 'chair', 'cup', 'bench', 'traffic_light', 'bicycle',
                        'bus/truck', 'car', 'watercraft', 'bird']
        cls_stat = dict()
        for c in cls_list:
            cls_stat[c] = 0
        for file in tqdm(files):
            tree = ET.parse(os.path.join('coco/Annotations/train_vidor', file.strip()+'.xml'))
            objs = tree.findall('object')
            if len(objs) > 1:
                new_files.append(file)
                continue
            if len(objs) <= 1:
                name = objs[0].find('name').text
                if name not in filter_clss:
                    new_files.append(file)
                cls_stat[name] += 1
                num += 1

        with open('coco/ImageSets/Main/train_vidor_new.txt', 'w') as f:
            for file in new_files:
                f.writelines(file+'\n')


def crop_traj_bboximgs(ext=False):
    dbname = args.imds
    if not ext:
        savedir = 'track_metric_learn/{:s}'.format(dbname)
    else:
        savedir = 'track_metric_learn/{:s}_ext'.format(dbname)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    print('Generate bbox images for {:s} tracking'.format(dbname))
    if dbname == 'vidvrd':
        modes = ['train', 'test']
        clss = ["turtle", "antelope", "bicycle", "lion", "ball", "motorcycle", "cattle", "airplane", "red_panda",
                "horse", "watercraft", "monkey", "fox", "elephant", "bird", "sheep", "frisbee", "giant_panda",
                "squirrel", "bus", "bear", "tiger", "train", "snake", "rabbit", "whale", "sofa", "skateboard", "dog",
                "domestic_cat", "person", "lizard", "hamster", "car", "zebra"]
        clscode2clsname = {
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
        thresh_num = 10000
        thresh_ext = 10000
    else:
        modes = ['train']
        clss = ['bread', 'cake', 'dish', 'fruits', 'vegetables', 'backpack', 'camera',
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
                'squirrel', 'tiger', 'adult', 'baby', 'child']
        clscode2clsname = {
            'n07745940': 'fruits', 'n07768694': 'fruits', 'n07753275': 'fruits', 'n07747607': 'fruits',
            'n07749582': 'fruits', 'n07753592': 'fruits', 'n07739125': 'fruits', 'n07753113': 'fruits', 'n07714571': 'vegetables',
            'n02769748': 'backpack', 'n03642806': 'laptop', 'n02802426': 'ball/sports_ball', 'n03134739': 'ball/sports_ball',
            'n03445777': 'ball/sports_ball', 'n03942813': 'ball/sports_ball', 'n04118538': 'ball/sports_ball',
            'n02799071': 'ball/sports_ball', 'n04254680': 'ball/sports_ball', 'n04409515': 'ball/sports_ball',
            'n04540053': 'ball/sports_ball', 'n04039381': 'racket', 'n04228054': 'ski', 'n03001627': 'chair',
            'n03797390': 'cup', 'n03271574': 'electric_fan', 'n03761084': 'microwave', 'n04070727': 'refrigerator',
            'n03211117': 'screen/monitor', 'n04256520': 'sofa', 'n04379243': 'table', 'n03467517': 'guitar',
            'n03928116': 'piano', 'n02828884': 'bench', 'n06874185': 'traffic_light', 'n02691156': 'aircraft',
            'n02834778': 'bicycle', 'n02924116': 'bus/truck', 'n02958343': 'car', 'n03790512': 'motorcycle',
            'n04468005': 'train', 'n04530566': 'watercraft', 'n01503061': 'bird', 'n01726692': 'snake',
            'n01662784': 'turtle', 'n02419796': 'antelope', 'n02131653': 'bear', 'n02437136': 'camel',
            'n02121808': 'cat', 'n02402425': 'cattle/cow', 'n02084071': 'dog', 'n02503517': 'elephant',
            'n02342885': 'hamster/rat', 'n02374451': 'horse', 'n02129165': 'lion', 'n02510455': 'panda',
            'n02324045': 'rabbit', 'n02411705': 'sheep/goat', 'n02355227': 'squirrel', 'n02129604': 'tiger',
            'n04591713': 'bottle', 'n04557648': 'bottle'
        }
        thresh_num = 15000
        thresh_ext = 15000

    for mode in modes:
        print('Start ', mode)
        if ext:
            print('Use ext ', ext)
        if mode == 'train':
            imglists = ['{:s}/ImageSets/Main/train.txt'.format(dbname),
                        'imagenet_det/ILSVRC2015/ImageSets/DET/train_{:s}.txt'.format(dbname),
                        'coco/ImageSets/Main/train_{:s}.txt'.format(dbname)
                       ]
            if ext:
                imglists = imglists[::-1]
        else:
            imglists = ['{:s}/ImageSets/Main/{:s}.txt'.format(dbname, mode)]

        clss_stat = dict(zip(clss, range(len(clss))))
        ext_stat = dict(zip(clss, range(len(clss))))  
        tmp = []
        for imglist in imglists:
            imglist_name = imglist.split('/')[0]
            print('Start', imglist_name)

            if imglist_name == 'vidvrd' or imglist_name == 'vidor':
                is_origin = True
            else:
                is_origin = False

            with open(imglist, 'r') as f:
                imgs = [im.strip() for im in f.readlines()]
            for img in tqdm(imgs):
                if imglist_name == 'coco':
                    xmlfile = 'coco/Annotations/train_{:s}/{:s}.xml'.format(dbname, img.zfill(12))
                    imgfile = 'coco/train2017/{:s}.jpg'.format(img.zfill(12))
                elif imglist_name == 'imagenet_det':
                    # imagenet_det/ILSVRC2015/Annotations/DET/train_vidvrd
                    year_set = img.split('/')[0].split('_')[0]
                    if '2013' in year_set:
                        xmlfile = 'imagenet_det/ILSVRC2015/Annotations/DET/train_{:s}/{:s}/{:s}/{:s}.xml' \
                            .format(dbname, img.split('/')[0], img.split('/')[1], img.split('/')[2])
                        imgfile = 'imagenet_det/ILSVRC2015/Data/DET/train/{:s}/{:s}/{:s}.JPEG' \
                            .format(img.split('/')[0], img.split('/')[1], img.split('/')[2])
                    else:  # 2014
                        xmlfile = 'imagenet_det/ILSVRC2015/Annotations/DET/train_{:s}/{:s}/{:s}.xml' \
                            .format(dbname, img.split('/')[0], img.split('/')[1])
                        imgfile = 'imagenet_det/ILSVRC2015/Data/DET/train/{:s}/{:s}.JPEG' \
                            .format(img.split('/')[0], img.split('/')[1])
                else:
                    if dbname == 'vidvrd':
                        vid = img.split('-')[0]
                        fid = img.split('-')[1]
                        xmlfile = '{:s}/Annotations/{:s}/{:s}/{:d}.xml'.format(dbname, mode, vid, int(fid))
                        imgfile = '{:s}/JPEGImages/{:s}/{:s}.jpg'.format(dbname, vid, fid)
                    else:
                        vid = img.split('-')[0]
                        subdir = img.split('-')[1]
                        fid = img.split('-')[2]
                        xmlfile = '{:s}/Annotations/{:s}/{:s}/{:s}/{:d}.xml'.format(dbname, mode, subdir, vid, int(fid))
                        imgfile = '{:s}/JPEGImages/{:s}/{:s}/{:s}.jpg'.format(dbname, subdir, vid, fid)

                tree = ET.parse(xmlfile)
                objs = tree.findall('object')
                if len(objs) == 0:
                    continue
                # im_h = int(tree.find('size').find('height').text)
                # im_w = int(tree.find('size').find('width').text)
                imgdata = cv2.imread(imgfile)

                for i, obj in enumerate(objs):
                    if imglist_name == 'imagenet_det':
                        #if obj.find('name').text.lower().strip() not in clscode2clsname:
                        #    continue
                        name = clscode2clsname[obj.find('name').text.lower().strip()]
                    else:
                        name = obj.find('name').text
                    #if name in tmp:
                    #    continue

                    if not ext:
                        clss_stat[name] += 1
                        if clss_stat[name] > thresh_num:
                            continue
                    else:   
                        if not is_origin:  # COCO or ImageNet DET
                            ext_stat[name] += 1
                            if ext_stat[name] > thresh_ext:
                                continue
                        else:
                            clss_stat[name] += 1
                            if clss_stat[name] > thresh_num:
                                continue
                    if name not in tmp:
                        tmp.append(name)
                        print(len(tmp), name)
                    bndbox = obj.find('bndbox')
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                    if not os.path.exists(savedir+'/'+mode+'/'+name):
                        os.makedirs(savedir+'/'+mode + '/' + name)
                    if xmin >= xmax or ymin >= ymax:
                        continue
                    cropbbox = imgdata[ymin:ymax, xmin:xmax, :]
                    savebbox = cv2.resize(cropbbox, dsize=(64, 128))
                    savepath = savedir+'/'+mode + '/' + name + '/' + img.split('/')[-1] + '-' + str(i) + '.jpg'
                    cv2.imwrite(savepath, savebbox)
            if is_origin:
                with open('tmp.pkl', 'wb') as f:
                    pkl.dump(ext_stat, f)
        for c in clss_stat.keys():
            print(c, clss_stat[c], ext_stat[c])


def tmp():
    with open('vidvrd/ImageSets/Main/test_full.txt', 'r') as f:
        fulls = [img.strip() for img in f.readlines()]
    with open('vidvrd/ImageSets/Main/test_filter.txt', 'r') as f:
        filter = [img.strip() for img in f.readlines()]
    imglist = []
    for im in tqdm(fulls):
        if im not in filter:
            imglist.append(im)
    with open('vidvrd/ImageSets/Main/test.txt', 'w') as f:
        for im in imglist:
            f.writelines(im+'\n')


def ext_vid():
    if args.imds == 'vidor':
        ext_clss = ['airplane', 'snake', 'turtle', 'antelope', 'bear',
                    'lion', 'giant_panda', 'squirrel', 'tiger']

        with open('vidvrd/ImageSets/Main/train.txt', 'r') as f:
            imgs = [im.strip() for im in f.readlines()]
        new_imgs = []
        ext_num = 0
        for img in tqdm(imgs):
            vid = img.split('-')[0]
            fid = img.split('-')[1]
            tree = ET.parse(os.path.join('vidvrd/Annotations/train', vid, str(int(fid))+'.xml'))
            root = tree.getroot()
            for obj in tree.findall('object'):
                if obj.find('name').text not in ext_clss:
                    root.remove(obj)
            objs = tree.findall('object')
            if not len(objs):
                continue
            else:
                ext_num += 1
                new_imgs.append(img)
                for obj in objs:
                    name = obj.find('name').text
                    if name == 'giant_panda':
                        obj.find('name').text = 'panda'
                    if name == 'airplane':
                        obj.find('name').text = 'aircraft'            
            new_dir = 'vidor/vidvrd/Annotations/{:s}/'.format(vid)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            tree.write(new_dir+fid+'.xml',encoding='utf-8', xml_declaration=True)
            
            impath = 'vidvrd/JPEGImages/{:s}/{:s}.jpg'.format(vid, fid)
            newdir = 'vidor/vidvrd/JPEGImages/{:s}'.format(vid)
            if not os.path.exists(newdir):
                os.makedirs(newdir)
            os.system('cp '+impath+' '+newdir)
        with open('vidor/vidvrd/train.txt', 'w') as f:
            for im in new_imgs:
                f.writelines(im+'\n')
        print('ext num from vidvrd', ext_num)
        

def vidor_human_split(mode='val'):
    # this function is used to split adult/child/baby/toy from the whole set
    # we name the two subset as db1 and db2
    with open('vidor/ImageSets/Main/%s.txt'%mode, 'r') as f:
        imglist = [im.strip() for im in f.readlines()]
    d1_list = []
    d2_list = []
    for im in tqdm(imglist):
        vid = im.split('-')[0]
        sub = im.split('-')[1]
        fid = int(im.split('-')[2])

        tree = ET.parse('vidor/Annotations/%s/%s/%s/%d.xml'%(mode, sub, vid, fid))
        objs = tree.findall('object')
        iw, ih = int(tree.find('size').find('width').text), int(tree.find('size').find('height').text)
        d1_bboxes, d1_names, d1_ges, d1_tids = [], [], [], []
        d2_bboxes, d2_names, d2_ges, d2_tids = [], [], [], []
        for obj in objs:
            bbox, name, tid = obj.find('bndbox'), obj.find('name').text, int(obj.find('trackid').text)
            xmin, ymin, xmax, ymax = int(bbox.find('xmin').text), int(bbox.find('ymin').text), \
                                    int(bbox.find('xmax').text), int(bbox.find('ymax').text)
            if name in ['adult', 'child', 'baby', 'toy']:
                d1_bboxes.append([xmin, ymin, xmax, ymax])
                d1_names.append(name)
                d1_ges.append(int(obj.find('generated').text))
                d1_tids.append(tid)
            else:
                d2_bboxes.append([xmin, ymin, xmax, ymax])
                d2_names.append(name)
                d2_ges.append(int(obj.find('generated').text))
                d2_tids.append(tid)
        if len(d1_bboxes) > 0:
            d1_list.append(im)
            dom = make_dom(sub, fid, ih, iw, d1_bboxes, d1_tids, d1_names, d1_ges)
            xml_dir = 'vidor/vidor_split/Annotations/%s/d1/%s/%s/'%(mode, sub, vid)
            if not os.path.exists(xml_dir):
                os.makedirs(xml_dir)
            xml_name = os.path.join(xml_dir, str(fid) + '.xml')
            if not os.path.isfile(xml_name):      
                print('d1', xml_name)
                with open(xml_name, 'wb') as f:
                    f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))
        if len(d2_bboxes) > 0:
            d2_list.append(im)
            dom = make_dom(sub, fid, ih, iw, d2_bboxes, d2_tids, d2_names, d2_ges)
            xml_dir = 'vidor/vidor_split/Annotations/%s/d2/%s/%s/'%(mode, sub, vid)
            if not os.path.exists(xml_dir):
                os.makedirs(xml_dir)
            xml_name = os.path.join(xml_dir, str(fid) + '.xml')
            if not os.path.isfile(xml_name):      
                print('d2', xml_name)
                with open(xml_name, 'wb') as f:
                    f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))

    with open('vidor/vidor_split/ImageSets/Main/%s_d1.txt'%mode, 'w') as f:
        for im in d1_list:
            f.writelines(im+'\n')
    with open('vidor/vidor_split/ImageSets/Main/%s_d2.txt'%mode, 'w') as f:
        for im in d2_list:
            f.writelines(im+'\n')


def tmp():
    for mode in ['test']:
        vid_list = []
        for sub in os.listdir('vidor/test/video'.format(mode)):
            for vid in os.listdir('vidor/test/video/'+sub):
                vid_list.append(sub+'-'+vid[:-4])
        with open('vidor/ImageSets/Main/{:s}_vids.txt'.format(mode), 'w') as f:
            for vid in vid_list:
                f.writelines(vid+'\n')
    for mode in []:
        vid_list = []
        for vid in os.listdir('vidvrd/annotation/{:s}/'.format(mode)):
            vid_list.append(vid[:-5])
        with open('vidvrd/ImageSets/Main/{:s}_vids.txt'.format(mode), 'w') as f:
            for vid in vid_list:
                f.writelines(vid+'\n')


if __name__ == '__main__':
    # --- analysis ---
    if args.func == 'data_analysis':
        #data = statistic()
        #data_analysis(data)
        test_vid_len()
    elif args.func == 'cal_proposal_num':
        cal_proposal_num(dbname='vidor')
    # --- vidvrd and vidor ----
    elif args.func == 'get_frames':
        # read videos and write them into frame jpegs
        get_frames()
    elif args.func == 'get_xmls':
        # generate xml files as PASCAL VOC format
        get_xmls()
    elif args.func == 'get_spilit':
        # generate train/test/val.txt for vidvrd and vidor
        get_spilit()
    elif args.func == 'filter_toomany':
        # filter out training samples if it's too many for vidvrd and vidor
        filter_toomany()
        cal_proposal_num(dbname='vidor')
    # --- ImageNet Det ---
    elif args.func == 'regenerate':
        re_imageset()
    elif args.func == 'filter_useless_cls':
        # filter useless class for ImageNet Det 
        filter_useless_cls()
    # --- COCO ---
    elif args.func == 'format_coco':
        format_coco()
    elif args.func == 'crop_traj_bbox':
        crop_traj_bboximgs()
    elif args.func == 'crop_traj_bbox_ext':
        crop_traj_bboximgs(ext=True)
    # vidvrd for vidor or vidor for vidvrd
    elif args.func == 'ext_vid':
        ext_vid()
    # --- check and validate ---
    elif args.func == 'check':
        check_anno()
    elif args.func == 'delete':
        detele_imgs()
    # split adult/child/baby/toy from the whole training set of vidor
    elif args.func == 'vidor_human_split':
        vidor_human_split('val')
    else:
        tmp()
