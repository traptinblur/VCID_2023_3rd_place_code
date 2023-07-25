echo "train 224 input size model on fold 1"
CUDA_VISIBLE_DEVICES=0 python seg_main_finetune.py --config r50_224_fold1

echo "train 224 input size model on fold 2 and 3"
CUDA_VISIBLE_DEVICES=0 python seg_main_finetune.py --config r50_224_fold2-3

echo "train 224 input size model on fold 4"
CUDA_VISIBLE_DEVICES=0 python seg_main_finetune.py --config r50_224_fold4

echo "train 224 input size model on fold 5"
CUDA_VISIBLE_DEVICES=0 python seg_main_finetune.py --config r50_224_fold5

echo "train 384 input size model on all folds"
CUDA_VISIBLE_DEVICES=0 python seg_main_finetune.py --config r50_384_all

echo "train 576 input size model on fold 1 and 4"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=9971 \
    seg_main_finetune.py --config r152_576_fold1-4 --dist_eval

echo "train 576 input size model on fold 2, 3 and 5"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=9972 \
    seg_main_finetune.py --config r152_576_fold2-3-5 --dist_eval
