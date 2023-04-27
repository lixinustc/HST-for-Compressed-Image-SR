CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_HST.py \
    --gpus 4 \
    --save_every 5000 \
    --eval_every 5000 \
    --batch_size 4 \
    --ckpt_path checkpoint \
    --valid_path valid \
    --distributed \
    --use_ema