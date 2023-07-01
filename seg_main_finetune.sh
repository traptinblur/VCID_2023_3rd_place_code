echo "Distributed Segmentation Model Training"
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=. python -m torch.distributed.launch  --nproc_per_node=4 --master_port=9971 seg_main_finetune.py \
 --dist_eval