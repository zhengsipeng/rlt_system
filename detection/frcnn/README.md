# Faster RCNN Object Detection Model
## File Structure
``` txt
-- detection/frcnn
---- cfgs
---- model
---- scripts
---- train.py
---- eval.py
-- data
---- pretrained_model
------ resnet101_caffe.pth
```

The pretrained_model can be downloaded from:

https://pan.baidu.com/s/19wtPLBZHnHE6re_C0ka5EA, code: ubyb

## Prerequisites
The code is ran under python 3.7 and pytorch 1.0.0.

Install all the python dependencies using pip:
``` python
cd frcnn
pip install -r requirement.txt
```

## Compilation
Compile the cuda dependencies using following simple commands:
``` python
python setup.py build develop
```
It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. The default version is compiled with Python 3.7, please compile by yourself if you are using a different python version.

## Object Detection DataSets
You can use multiple datasets to extend your object detection dataset
``` text
-- ImageNet DET (200 class)
     It is saved in 'data/imagenet_det' as PASCAL VOC format, you can see specific class name list 
     in http://image-net.org/challenges/LSVRC/2015/browse-det-synsets, notice that ImageNet DET use
     symbol codes to replace class indeces, check http://image-net.org/challenges/LSVRC/2015/browse-det-synsets to see corresponding category of each symbol code.
-- COCO (80 class)
---- It is saved in 'data/coco', raw annotations in 'data/coco/annotation' and images in 'train2017'
-- VIDVRD
---- It is saved in 'data/vidvrd' with PASCAL VOC format
-- VIDOR
---- It is saved in 'data/vidor' with PASCAL VOC format
```
You can see more information about vidvrd and vidor in 'data/data_format' and there is a data processing script 'process.py' under 'data' dir.


## Data Preparation
I place all training data under 'data' dir, the data format of follows PASCAL VOC format, which is formulated as:
```
-- Annotations
---- /predir/imgid1.xml
---- /predir/imgid2.xml
-- ImageSets
---- Main
------ train.txt
------ val.txt
------ test.txt
-- JPEGImages
---- /predir/imgid1.jpg
---- /predir/imgid2.jpg
---- ...
```
If you are not familiar with PASCAL VOC format, Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets. Actually, you can refer to any others. After downloading the data, creat softlinks in the folder data/. I strongly recommond you to validate the code under PASCAL VOC dataset.

Also, I provide some processing code to generate training data in 'data/process.py'. Well, though it's specifically written for VIDVRD and VIDOR datasets, but you can still refer to it.


## Pretrained Model
We used two pretrained models in our experiments, VGG and ResNet101. You can download these two models from:

* VGG16: [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

* ResNet101: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

Download them and put them into the 'data/pretrained_model/'.

**NOTE**. We compare the pretrained models from Pytorch and Caffe, and surprisingly find Caffe pretrained models have slightly better performance than Pytorch pretrained. We would suggest to use Caffe pretrained models from the above link to reproduce our results.

**If you want to use pytorch pre-trained models, please remember to transpose images from BGR to RGB, and also use the same data transformer (minus mean and normalize) as used in pretrained model.**


## Train

Before training, set the right directory to save and load the trained models. Change the arguments "save_dir" and "load_dir" in train.py and eval.py to adapt to your environment.

To train a faster R-CNN model with res101 on pascal_voc, run the following command under the 'detection' dir:
```
bash frcnn/scripts/train.sh
```
you can set some training settings in 'train.sh', and some is important
```
--fpn # Feature Pyramid Network
--ls  # Using large scale
--mGPUs  # multiple GPUs
--bs  # batch size
--nw  # number of works
--r   # resume
```
Above, BATCH_SIZE and WORKER_NUMBER can be set adaptively according to your GPU memory size. **On Titan Xp with 12G memory, it can be up to 4**.

## Test
If you want to evlauate the detection performance of a pre-trained model, simply run
```
bash frcnn/scripts/eval.sh
```
Specify the specific model session, chechepoch and checkpoint, e.g., SESSION=1, EPOCH=6, CHECKPOINT=416.

I add a new script for evaluation with batchsize > 1, you can run by:
```
bash frcnn/scripts/eval_batch.sh
```

Pretrained models for VidOR and VidVRD has been provided, the mAP is 0.37 (vidor_ext, 80 categories) and 0.674 (VidVRD):

https://pan.baidu.com/s/19wtPLBZHnHE6re_C0ka5EA, code:ubyb

You should place the pratrained model under 'frcnn/models/res101/vidor/...' or 'frcnn/models/res101/vidvrd/...'

## Demo
[TODO]
If you want to run detection on your own images with a pre-trained model, download the pretrained model listed in above tables or train your own models at first, then add images to folder $ROOT/images, and then run
```
python demo.py --net vgg16 \
               --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
               --cuda --load_dir path/to/model/directoy
```

Then you will find the detection results in folder $ROOT/images.

## ToolBox
I put other preprocess and reprocess funcs in toolbox.py including xml writing, visualization and etc.
