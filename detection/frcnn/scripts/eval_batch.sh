CUDA_VISIBLE_DEVICES=4,5,6,7 python -m object_detection.eval_batch \
--dataset vidor_test \
--net res101 \
--checksession 1 \
--checkepoch 20 \
--checkpoint 20248 \
--cuda \
--mGPUs \
