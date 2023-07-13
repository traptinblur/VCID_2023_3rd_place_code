echo "Distributed Segmentation Model Training"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch  --nproc_per_node=4 --master_port=9971 \
    seg_main_finetune.py --config base_config --dist_eval
