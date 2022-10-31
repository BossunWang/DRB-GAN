CUDA_VISIBLE_DEVICES=0 python test.py \
    --test_dataset "../Stable_Diffusion_Demo" \
    --tgt_dataset "../art_dataset_v2" \
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
    --sample_dir "train_log_v3/stylized_stable_diffusion_demo_val" \
    --save_extend ".png" \
    --pretrain_model "train_log_v3/checkpoint_DRB_GAN_animal_style_data_v2/DRBGAN_it_699999.pt" \
    --mixture_list "LOUIS WAIN" "kaka" \
    --mixture_weights 0.5 0.5

CUDA_VISIBLE_DEVICES=0 python test.py \
    --test_dataset "../Stable_Diffusion_Demo" \
    --tgt_dataset "../art_dataset_v2" \
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
    --sample_dir "train_log_v3/stylized_stable_diffusion_demo_val_add_random_noise" \
    --save_extend ".png" \
    --pretrain_model "train_log_v3/checkpoint_DRB_GAN_animal_style_data_v2/DRBGAN_it_699999.pt" \
    --mixture_list "LOUIS WAIN" "kaka" \
    --mixture_weights 0.5 0.5 \
    --add_random_noise

#CUDA_VISIBLE_DEVICES=0 python test.py \
#    --test_dataset "/media/glory/Transcend/Dataset/stylegan_v2_data/cat-256x256-20220623T040454Z-001/cat-256x256/stylegan2-config-f-psi-0.5/000000" \
#    --tgt_dataset "../art_dataset_v2" \
#    --src_img_size 512 \
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
#    --sample_dir "train_log_v3/stylized_stylegan2_cat" \
#    --save_extend ".png" \
#    --pretrain_model "train_log_v3/checkpoint_DRB_GAN_animal_style_data_v2/DRBGAN_it_699999.pt" \
#    --mixture_list "LOUIS WAIN" "kaka" \
#    --mixture_weights 0.5 0.5 \
#    --sample_compared

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