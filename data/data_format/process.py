import cv2
import os
import json
import numpy as np
import argparse
import random
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

parser = argparse.ArgumentParser('Data_Process')
parser.add_argument('--func', dest='func', default='regenerate', type=str)
parser.add_argument('--imds', dest='imds', default='vidvrd', type=str)
parser.add_argument('--frame_gap', dest='frame_gap', default=8, type=int)
args = parser.parse_args()

global modes
if args.imds == 'vidvrd':
    modes = ['train', 'test']
elif args.imds == 'vidor':
    modes = ['train', 'val']
else:
    raise NotImplementedError


# ---------------------------------------
# read videos and write them into frames
# ---------------------------------------
def write_frames(vcap, videoname, video_out_path):
    frame_count = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
    cur_frame = 0
    while cur_frame < frame_count:
        suc, frame = vcap.read()
        if not suc:
            cur_frame += 1
            print("warning, %s frame of %s failed" % (cur_frame, videoname))
            continue
        if cur_frame % args.frame_gap != 0:
            cur_frame += 1
            continue
        im = frame.astype("float32")
        frame_file = os.path.join(video_out_path, "%05d.jpg" % cur_frame)
        cv2.imwrite(frame_file, im)
        cur_frame += 1


def generate_frames():
    if args.imds == 'vidvrd':
        for i, vid in enumerate(os.listdir(args.imds + '/video')):
            if i % 100 == 0:
                print(i)
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
            write_frames(vcap, videoname, video_out_path)
    elif args.imds == 'vidor':
        badfiles = []
        for subdir in os.listdir(args.imds + '/video'):
            subpath = args.imds + '/video/' + subdir
            for vid in os.listdir(subpath):
                videofile = subpath + '/' + vid
                vcap = cv2.VideoCapture(videofile)
                if not vcap.isOpened():
                    badfiles.append(vid)
                    print("cannot open %s" % videofile)
                    continue

                videoname = os.path.basename(videofile).split('.')[0]
                video_out_path = os.path.join('vidor/JPEGImages/' + subdir + '/', videoname)
                if os.path.exists(video_out_path):
                    continue
                if not os.path.exists(video_out_path):
                    os.makedirs(video_out_path)
                write_frames(vcap, videoname, video_out_path)
        with open('{:s}_badfiles.txt'.format(args.imds), 'w') as f:
            for badfile in badfiles:
                f.writelines(badfile+'/n')


# -------------------------------
# generate xml annotations
# -------------------------------
def make_xml(subdir, im_id, ih, iw, boxes, tids, clses, generated):
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
        dom = make_xml(subdir, num, ih, iw, boxes, tids, clss, generated)
        xml_name = os.path.join(xml_dir, str(num) + '.xml')
        with open(xml_name, 'wb') as f:
            f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))


def generate_xmls():
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


# -------------------------------------------
# filter out those frames with no candidates
# -------------------------------------------
def filter_out():
    if args.imds == 'vidvrd':
        for mode in modes:
            new_imagelist = []
            f = open('vidvrd/ImageSets/Main/{:s}.txt'.format(mode), 'r')
            imagelist = [image.strip() for image in f.readlines()]
            for image in imagelist:
                vid_name = image.split('-')[0]
                frame_id = int(image.split('-')[1])
                annofile = 'vidvrd/Annotations/{:s}/{:s}/{:d}.xml'.format(mode, vid_name, frame_id)
                if not os.path.exists(annofile):
                    continue
                tree = ET.parse(annofile)
                objs = tree.findall('object')
                if len(objs) == 0:
                    continue
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
            for image in imagelist:
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
                  'electric_fan', 'elephant', 'faucet', 'fish', 'frisbee', 'fruits', 'guitar', 'hamster/rat',
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


# ------------------------------------
# generate trainval.txt and test.txt
# ------------------------------------
def spilit_list():
    if args.imds == 'vidvrd':
        modesets = ['train', 'test']
    else:
        modesets = ['train', 'val']
    for mode in modesets:
        rootdir = args.imds+'/ImageSets/Main/'
        if not os.path.exists(rootdir):
            os.makedirs(rootdir)
        f = open(rootdir+mode+'.txt', 'w')
        frame_list = list()
        if args.imds == 'vidvrd':
            for i, vid in enumerate(os.listdir('vidvrd/annotation/{:s}'.format(mode))):
                print(i)
                for frame in os.listdir('vidvrd/JPEGImages/'+vid.split('.')[0]):
                    frame_id = vid.split('.')[0]+'-'+frame.split('.')[0]
                    frame_list.append(frame_id)
        elif args.imds == 'vidor':
            for i, subdir in enumerate(os.listdir('vidor/annotation/{:s}'.format(mode))):
                print(i)
                for vid in os.listdir('vidor/annotation/{:s}/'.format(mode)+subdir):
                    for frame in os.listdir('vidor/JPEGImages/'+subdir+'/'+vid.split('.')[0]):
                        frame_id = vid.split('.')[0] + '-' + subdir + '-' + frame.split('.')[0]
                        frame_list.append(frame_id)
        else:
            raise NotImplementedError

        random.shuffle(frame_list)

        for frame_id in frame_list:
            f.writelines(frame_id+'\n')
        f.close()


# ------------------------------------------------------------
# regenerate ImageSets train_new.txt for ImageNet-DET dataset
# some missing and mistakes exist in raw annotations
# Also, we only select object categories for vidor and vidvrd
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

        for subdir in os.listdir(rootdir):
            print(subdir)
            filenum = 0
            if subdir == 'ILSVRC2013_train': # imagenet_det_2013
                for cls in os.listdir(rootdir+'/'+subdir):
                    if cls not in cls_indexs:
                        continue
                    for xmlfile in os.listdir(rootdir+'/'+subdir+'/'+cls):
                        imgid = xmlfile.split('.')[0]
                        imageset_list.append(rootdir+'/'+subdir+'/'+cls+'/'+imgid)
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
                        print(233)
                        imageset_list.append(rootdir+'/'+subdir+'/'+filename)
                        filenum += 1
        with open('imagenet_det/ILSVRC2015/ImageSets/DET/train_new.txt', 'w') as f:
            for image in imageset_list:
                f.writelines(image+'\n')
        print(filenum)

    elif args.imds == 'vidor':
        image_indexs = []
    else:
        raise NotImplementedError


if __name__ == '__main__':
    if args.func == 'get_frames':
        generate_frames()
    elif args.func == 'get_xmls':
        generate_xmls()
    elif args.func == 'data_analyse':
        data = statistic()
        data_analysis(data)
    elif args.func == 'split':
        spilit_list()
    elif args.func == 'filter':
        filter_out()
    elif args.func == 'regenerate':
        re_imageset()
    else:
        raise NotImplementedError
