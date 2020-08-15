CUDA_VISIBLE_DEVICES=0,1,2,3 python -m object_detection.get_bbox_feat \
--dataset vidor_ext \
--net res101 \
--checksession 1 \
--checkepoch 20 \
--checkpoint 20248 \
--cuda \
--mGPUs \

