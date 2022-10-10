#CUDA_VISIBLE_DEVICES=1 python test.py \
#    --test_dataset "../../Animal_dataset/val/afhg_val/cat" \
#    --tgt_dataset "../data_art_backup" \
#    --tgt_img_size 512 \
#    --tgt_crop_size 512 \
#    --workers 1 \
#    --feature_dim 639488 \
#    --gamma_dim 200 \
#    --beta_dim 200 \
#    --omega_dim 4 \
#    --encoder_out_ch 128 \
#    --db_number 4 \
#    --ws 128 \
#    --M 2 \
#    --sample_dir "train_log_v2/stylized_afhg_cat_val" \
#    --ref_style_dir "train_log_v2/style_ref_val" \
#    --stored_content_dir "train_log_v2/afhq_cat_val" \
#    --save_extend ".png" \
#    --pretrain_model "train_log_v2/checkpoint_DRB_GAN_animal/DRBGAN_it_599999.pt" \
#    --mixture_list "LOUIS WAIN" "Ukiyo_e" \
#    --mixture_weights 0.5 0.5


CUDA_VISIBLE_DEVICES=1 python test.py \
    --test_dataset "../photo2fourcollection-20210312T150938Z-001/photo2fourcollection/testA" \
    --tgt_dataset "../data_art_backup" \
    --tgt_img_size 512 \
    --tgt_crop_size 512 \
    --workers 1 \
    --feature_dim 639488 \
    --gamma_dim 200 \
    --beta_dim 200 \
    --omega_dim 4 \
    --encoder_out_ch 128 \
    --db_number 4 \
    --ws 128 \
    --M 2 \
    --sample_dir "train_log_v2/stylized_photo_scene_val" \
    --ref_style_dir "train_log_v2/style_ref_photo_scene_val" \
    --stored_content_dir "train_log_v2/photo_scene_val" \
    --save_extend ".png" \
    --pretrain_model "train_log_v2/checkpoint_DRB_GAN_animal/DRBGAN_it_599999.pt" \
    --mixture_list "LOUIS WAIN" "Ukiyo_e" \
    --mixture_weights 0.5 0.5
