CUDA_VISIBLE_DEVICES=1,2,3,4,5 python -m frcnn.scripts.train \
--dataset vidor_ext \
--net res101 \
--epochs 20 \
--save_dir frcnn/models \
--bs 10 \
--nw 5 \
--cuda \
--ls \
--mGPUs \
