CUDA_VISIBLE_DEVICES=1,0 python train.py \
    --src_dataset '../../AnimeGAN/dataset/train_photo/' \
    --tgt_dataset '../../AnimeGAN/dataset/Shinkai_new/style' \
    --tgt_smooth_dataset '../../AnimeGAN/dataset/Shinkai_new/smooth' \
    --val_dataset '../../AnimeGAN/dataset/val/' \
    --init_epoch 10 \
    --epoch 200 \
    --print_freq 100 \
    --save_freq 2 \
    --d_adv_weight 1. \
    --g_adv_weight 5. \
    --con_weight 1.2 \
    --sty_weight 2. \
    --color_weight 10. \
    --tv_weight 1.
#    --pretrained True \
#    --pretrain_model 'checkpoint/AnimeGAN_Epoch_8.pt'
