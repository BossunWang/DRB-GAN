CUDA_VISIBLE_DEVICES=0 python style_classify.py \
    --tgt_dataset "../../data_art_backup" \
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
    --pretrain_model "../train_log_v2/checkpoint_DRB_GAN_animal/DRBGAN_it_599999.pt" \
    --incorrect_dir "incorrect_style_prediction_v2" \
    --correct_dir "../../art_dataset_v2"


