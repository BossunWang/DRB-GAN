#CUDA_VISIBLE_DEVICES=0 python test.py \
#    --test_dataset "/media/glory/Transcend/Dataset/Scene/DIV2K_valid_HR" \
#    --tgt_dataset "../art_dataset_v2" \
#    --src_img_size 512 \
#    --tgt_img_size 256 \
#    --tgt_crop_size 256 \
#    --workers 1 \
#    --feature_dim 163840 \
#    --gamma_dim 64 \
#    --beta_dim 64 \
#    --omega_dim 64 \
#    --encoder_out_ch 64 \
#    --db_number 8 \
#    --ws 128 \
#    --M 0 \
#    --assigned_labels 6 \
#    --sample_dir "train_log_v4/stylized_DIV2K_valid" \
#    --save_extend ".png" \
#    --pretrain_model "train_log_v4/checkpoint_DRB_GAN_edge_patch_style_data_v2/DRBGAN_it_499999.pt" \
#    --mixture_list "LOUIS WAIN" "kaka" \
#    --mixture_weights 0.5 0.5 \
#    --sample_compared
#
#CUDA_VISIBLE_DEVICES=0 python test.py \
#    --test_dataset "../../Animal_dataset/val/afhg_val/cat" \
#    --tgt_dataset "../art_dataset_v2" \
#    --tgt_img_size 256 \
#    --tgt_crop_size 256 \
#    --workers 1 \
#    --feature_dim 163840 \
#    --gamma_dim 64 \
#    --beta_dim 64 \
#    --omega_dim 64 \
#    --encoder_out_ch 64 \
#    --db_number 8 \
#    --ws 128 \
#    --M 0 \
#    --assigned_labels 6 \
#    --sample_dir "train_log_v4/stylized_afhg_val_cat" \
#    --ref_style_dir "train_log_v4/style_ref_afhg_val_cat" \
#    --stored_content_dir "train_log_v4/afhq_cat_val" \
#    --save_extend ".png" \
#    --pretrain_model "train_log_v4/checkpoint_DRB_GAN_edge_patch_style_data_v2/DRBGAN_it_499999.pt" \
#    --mixture_list "LOUIS WAIN" "kaka" \
#    --mixture_weights 0.5 0.5

#CUDA_VISIBLE_DEVICES=0 python test.py \
#    --test_dataset "../Stable_Diffusion_Demo" \
#    --tgt_dataset "../art_dataset_v2" \
#    --tgt_img_size 256 \
#    --tgt_crop_size 256 \
#    --workers 1 \
#    --feature_dim 163840 \
#    --gamma_dim 64 \
#    --beta_dim 64 \
#    --omega_dim 64 \
#    --encoder_out_ch 64 \
#    --db_number 8 \
#    --ws 128 \
#    --M 0 \
#    --assigned_labels 6 \
#    --sample_dir "train_log_v4/stylized_Stable_Diffusion_Demo" \
#    --save_extend ".png" \
#    --pretrain_model "train_log_v4/checkpoint_DRB_GAN_edge_patch_style_data_v2/DRBGAN_it_499999.pt" \
#    --mixture_list "LOUIS WAIN" "kaka" \
#    --mixture_weights 0.5 0.5 \
#    --sample_compared

CUDA_VISIBLE_DEVICES=0 python test.py \
    --test_dataset "../Stable_Diffusion_Demo" \
    --tgt_dataset "../art_dataset_v2" \
    --tgt_img_size 256 \
    --tgt_crop_size 256 \
    --workers 1 \
    --feature_dim 163840 \
    --gamma_dim 64 \
    --beta_dim 64 \
    --omega_dim 64 \
    --encoder_out_ch 64 \
    --db_number 8 \
    --ws 128 \
    --M 0 \
    --assigned_labels 6 \
    --sample_dir "train_log_v4/stylized_Stable_Diffusion_Demo_add_random_noise" \
    --save_extend ".png" \
    --pretrain_model "train_log_v4/checkpoint_DRB_GAN_edge_patch_style_data_v2/DRBGAN_it_499999.pt" \
    --mixture_list "LOUIS WAIN" "kaka" \
    --mixture_weights 0.5 0.5 \
    --add_random_noise