CUDA_VISIBLE_DEVICES=1 python fid.py \
    --paths "../../train_log_v2/style_ref_val" "../../train_log_v2/stylized_afhg_cat_val" \
    --img_size 512

CUDA_VISIBLE_DEVICES=1 python fid.py \
    --paths "../../train_log_v2/style_ref_photo_scene_val" "../../train_log_v2/stylized_photo_scene_val" \
    --img_size 256