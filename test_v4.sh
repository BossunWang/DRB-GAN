CUDA_VISIBLE_DEVICES=0 python test.py \
    --test_dataset "/media/glory/Transcend/Dataset/Scene/DIV2K_valid_HR" \
    --tgt_dataset "../art_dataset_v3" \
    --src_img_size 512 \
    --tgt_img_size 512 \
    --tgt_crop_size 512 \
    --workers 1 \
    --feature_dim 655360 \
    --gamma_dim 128 \
    --beta_dim 128 \
    --omega_dim 4 \
    --encoder_out_ch 128 \
    --db_number 4 \
    --ws 128 \
    --M 2 \
    --assigned_labels 5 \
    --sample_dir "train_log_v4/stylized_DIV2K_valid_" \
    --save_extend ".png" \
    --pretrain_model "train_log_v4/checkpoint_DRB_GAN_edge_patch_animal_scene_style_data_v3/DRBGAN_it_499999.pt" \
    --mixture_list "LOUIS WAIN" "kaka" \
    --mixture_weights 0.5 0.5 \
    --sample_compared

#CUDA_VISIBLE_DEVICES=0 python test.py \
#    --test_dataset "../../Animal_dataset/val/afhg_val/cat" \
#    --tgt_dataset "../art_dataset_v2" \
#    --tgt_img_size 512 \
#    --tgt_crop_size 512 \
#    --workers 1 \
#    --feature_dim 655360 \
#    --gamma_dim 128 \
#    --beta_dim 128 \
#    --omega_dim 4 \
#    --encoder_out_ch 128 \
#    --db_number 4 \
#    --ws 128 \
#    --M 2 \
#    --sample_dir "train_log_v3/stylized_afhg_val_cat" \
#    --ref_style_dir "train_log_v3/style_ref_afhg_val_cat" \
#    --stored_content_dir "train_log_v3/afhq_cat_val" \
#    --save_extend ".png" \
#    --pretrain_model "train_log_v3/checkpoint_DRB_GAN_animal_style_data_v2/DRBGAN_it_699999.pt" \
#    --mixture_list "LOUIS WAIN" "kaka" \
#    --mixture_weights 0.5 0.5