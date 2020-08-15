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
import cv2
import glob
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from model.utils.net_utils import vis_detections, vis_detections_allclss


def get_vidor_clss():
    vidor_clss = ['__background__',
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
                ]
    return vidor_clss

mega_clss = ['__background__',  # always index 0
            'airplane', 'antelope', 'bear', 'bicycle',
            'bird', 'bus', 'car', 'cattle',
            'dog', 'domestic_cat', 'elephant', 'fox',
            'giant_panda', 'hamster', 'horse', 'lion',
            'lizard', 'monkey', 'motorcycle', 'rabbit',
            'red_panda', 'sheep', 'snake', 'squirrel',
            'tiger', 'train', 'turtle', 'watercraft',
            'whale', 'zebra']
vidor_clss2id = dict(zip(get_vidor_clss(), range(len(vidor_clss))))
mega_clss2id = dict(zip(mega_clss, range(len(vidor_clss))))


def make_xml(im_id, ih, iw, boxes, clss, confs):
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


def write_xmls(ori_bboxes, fid, xml_path, ih, iw, dbtype='vidor'):
    clss, boxes, confs = [], [], []
    for i in range(len(ori_bboxes)):
        boxes.append(ori_bboxes[i, :4])
        if dbtype == 'vidor':
            clss.append(vidor_clss[int(ori_bboxes[i, 5])])
        else:
            clss.append(mega_clss[int(ori_bboxes[i, 5])])
        confs.append(ori_bboxes[i, 4])
        dom = make_xml(fid, ih, iw, boxes, clss, confs)
        with open(xml_path, 'wb') as f:
            f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))


def det_to_xml():
    # write frcnn det results to xml dir
    max_per_image = 100
    det_dir = '../output/res101/vidor_2020_test/'
    det_files = glob.glob(det_dir+'*/*.pkl')
    neg, pos = 0,0
    for det_file in tqdm(det_files):
        if det_file == '../output/res101/vidor_2020_test/faster_rcnn_10/detections.pkl':
            continue
        sub, vid = det_file[:-4].split('/')[-2:]
        xml_dir = '../../data/vidor/track_bbox/test/%s/%s/'%(sub, vid)
        if not os.path.exists(xml_dir):
            os.makedirs(xml_dir)
        cap = cv2.VideoCapture('../../data/vidor/test/video/%s/%s.mp4'%(sub, vid))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        with open(det_file, 'rb') as f:
            queue_dict = pkl.load(f)
        for fid, bboxes in queue_dict.items():
            xmlpath = xml_dir+'%s.xml'%fid[:-4]
            if bboxes.shape[0] == 0 or os.path.isfile(xmlpath):
                continue
            image_scores = bboxes[:, 4]
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                keep = np.where(bboxes[:, -1] >= image_thresh)[0]           
                bboxes = bboxes[keep, :]
            write_xmls(bboxes, fid[:-4], xmlpath, height, width)


def det_mega_to_xml():
    # write mega det results to xml dir
    max_per_image = 100
    with open('../../data/vidor/track_bbox/mega_predictions.pkl', 'rb') as f:
        preds = pkl.load(f)

    with open('../../data/vidor/test/test.txt', 'r') as f:
        test_imgs = [im.strip() for im in f.readlines()]  # vid-sub-fid
    for image_id, pred in tqdm(enumerate(preds)):
        vid, sub, fid = test_imgs[image_id].split('-')
        xmldir = '../../data/vidor/track_bbox/test_mega/%s/%s/'%(sub, vid)
        if not os.path.exists(xmldir):
            os.makedirs(xmldir)
        xmlpath = xmldir+'%s.xml'%fid
        im = cv2.imread('../../data/vidor/test/JPEGImages/%s/%s/%s.jpg'%(sub, vid, fid))
        ih, iw, _ = im.shape
        pred_boxlist = pred.resize((iw, ih))
        bboxes = pred_boxlist.bbox
        scores = pred_boxlist.get_field('scores')
        labels = pred_boxlist.get_field('labels')
        dets = []
        for i in range(len(bboxes)):
            dets.append(bboxes[i].tolist()+[scores[i].tolist(), labels[i].tolist()])
        dets = np.asarray(dets)
        image_scores = dets[:, -1]
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            keep = np.where(bboxes[:, -1] >= image_thresh)[0]           
            bboxes = bboxes[keep, :]
        write_xmls(dets, fid[:-4], xmlpath, ih, iw, dbtype='mega')



def vis(dbname='vidor'):
    # vis det results in xml files
    xml_files = glob.glob('../../data/vidor/track_bbox/test/0032/*/*xml')[:100]
    for xml_file in tqdm(xml_files):
        dets = []
        sub, vid, fid = xml_file[:-4].split('/')[-3:]
        im_file = '../../data/vidor/test/JPEGImages/%s/%s/%s.jpg'%(sub, vid, fid)
        tree = ET.parse(xml_file)
        objs = tree.findall('object')
        for obj in objs:
            if dbname == 'vidor':
                label = vidor_clss2id[obj.find('name').text]
                clss_name = vidor_clss
            else:
                label = mega_clss2id[obj.find('name').text]
                clss_name = mega_clss
            bndbox = obj.find('bndbox')
            xmin, ymin = float(bndbox.find('xmin').text), float(bndbox.find('ymin').text)
            xmax, ymax = float(bndbox.find('xmax').text), float(bndbox.find('ymax').text)
            conf = float(bndbox.find('confidence').text)
            dets.append([xmin, ymin, xmax, ymax, conf, label])
        dets = dets[np.argsort(np.asarray(dets)[:, 4])[::-1]]
        im2show = vis_detections_allclss(cv2.imread(im_file), dets[:10], clss_name, 0.05)

        if not os.path.exists('../../data/vis/vidor_test/%s/%s/'%(sub, vid)):
            os.makedirs(save_dir)    

        cv2.imwrite(save_dir+'%d.jpg'%int(fid), im2show)


import json
def search_sub(vid):
    vpaths = glob.glob('../../data/vidor/test/video/*/*.mp4')
    for vpath in vpaths:
        if vid == vpath[:-4].split('/')[-1]:
            return vpath[:-4].split('/')[-2]
    raise IOError


def get_trajs_fromxml():
    rootvidor = '../../data/vidor/annotation/%s/'%split
    vlist = []
    for sub in os.listdir(rootvidor):
        for vid in os.listdir(rootvidor+sub):
            vlist.append(rootvidor+sub+'/'+vid)

    all_trajs = {}
    t_num = 0
    for vfile in tqdm(vlist):
        with open(vfile, 'r') as f:
            anno = json.load(f)
        vid = vfile.split('/')[-1][:-5]
        so_inst = anno['subject/objects']
        all_trajs[subvid] = dict()
        for traj in so_inst:
            all_trajs[vid][traj['tid']] = {'category': traj['category'], 'trajectory': []}
            t_num += 1
        trajs = anno['trajectories']
        for fid, bboxes in enumerate(trajs):
            for bbox in bboxes:
                coords = [bbox['bbox']['xmin'], bbox['bbox']['ymin'], \
                        bbox['bbox']['xmax'], bbox['bbox']['ymax'], fid]
                all_trajs[subvid][bbox['tid']]['trajectory'].append(coords)

    print('Trajectory number:', t_num)
    with open('cache/vidor_%s_trajs.pkl'%split, 'wb') as f:
        pkl.dump(all_trajs, f)


def get_rcnn_imglist(split, mode):
    # generate rcnn imglist from traj file for feature extraction
    with open('cache/vidor_%s_trajs.pkl'%split, 'rb') as f:
        trajs = pkl.load(f)
    
    frames = dict()
    inum = 0
    for vid, vtrajs in tqdm(trajs.items()):
        sub, vid = subvid.split('-')
        if vid not in frames:
            frames[vid] = []
        for tid, traj in vtrajs.items():
            bboxes = traj['trajectory']
            tlen = len(bboxes)
            stt_fid = bboxes[min(tlen-1, 5)][4]
            end_fid = bboxes[max(-tlen+1, -5)][4]
            if stt_fid not in frames[subvid]:
                frames[subvid].append([stt_fid, tid]+bboxes[min(tlen-1, 5)][:4])
                inum+=1
            if end_fid not in frames[subvid]:
                frames[subvid].append([end_fid, tid]+bboxes[max(-tlen+1, -5)][:4])
                inum+=1
    print(inum)
    with open('cache/%s_rcnn.txt'%split, 'w') as f:
        for subvid, fs in frames.items():
            sub = search_sub(vid)
            for fidbbox in fs:
                fid = fidbbox[0]
                tid = fidbbox[1]
                box = [int(i) for i in fidbbox[2:]]
                f.writelines('%s/%s/%d/%d/%d/%d/%d/%d\n'%(sub, vid, fid, tid, box[0], box[1], box[2], box[3]))
        

def upload_miss_imgs():
    with open('cache/train_rcnn.txt', 'r') as f:
        imgs = [im.strip() for im in f.readlines()]
    for im in tqdm(imgs):
        sub, vid, fid = im.split('/')[:3]
        imfile = '/'.join((sub, vid, fid))+'.jpg'
        if not os.path.isfile('../../data/vidor/JPEGImages/'+imfile):
            command = 'scp -i ../scorpio_zsp_2220 -P 2220 root@202.112.113.15:/data4/zsp/video_relation/data/vidor/JPEGImages/%s ../data/vidor/JPEGImages/%s/%s'%(imfile, sub, vid)
            os.system(command)



def gather_rcnn_feat(split='val'):
    rcnn_feats = dict()  
    with open('vidor_%s_trajs.pkl'%split, 'rb') as f:
        trajs = pkl.load(f)
    
    for vid, vtrajs in tqdm(trajs.items()):
        sub = search_sub(sub)
        rcnn_feats[vid] = dict()
        for tid, traj in vtrajs.items():
            rcnn_feats[vid][tid] = dict()
            bboxes = traj['trajectory']
            tlen = len(bboxes)
            stt_fid = bboxes[min(tlen-1, 5)][4]
            stt_box = bboxes[min(tlen-1, 5)][:4]
            end_fid = bboxes[max(-tlen+1, -5)][4]
            end_box = bboxes[max(-tlen+1, -5)][:4]
            with open('cache/rcnn_feat/%s/%s.pkl'%(sub, vid), 'rb') as f:
                rcnn_feat = pkl.load(f)
            stt_feats = rcnn_feat[stt_fid]
            end_feats = rcnn_feat[end_fid]
            match_feat = []
            match_score = []
            for stt_feat in stt_feats:
                if stt_box[0] == stt_feat[0] and stt_box[1] == stt_feat[1] and stt_box[2] == stt_feat[2]:
                    match_feat.append(stt_feat[86:])
                    match_score.append(stt_feat[6:86])
            for end_feat in end_feats:
                if end_box[0] == end_feat[0] and end_box[1] == end_feat[1] and end_box[2] == end_feat[2]:
                    match_feat.append(end_feat[86:])
                    match_score.append(end_feat[6:86])
        
            match_feat = np.asarray(match_feat).reshape(2, 2048)
            match_score = np.asarray(match_score).reshape(2, 81)

            rcnn_feats[vid][tid]['rcnn_feat'] = match_feat
            rcnn_feats[vid][tid]['score'] = match_score


if __name__ == '__main__':
    '''
    toolbox
    # vis('mega')
    # det_to_xml()
    # det_mega_to_xml()
    # get_trajs_fromxml()
    # get_rcnn_imglist('mm19')
    # upload_miss_imgs()
    # gather_rcnn_feat()
    '''
    format_rcnn()