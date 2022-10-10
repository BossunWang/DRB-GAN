#CUDA_VISIBLE_DEVICES=1 python lpips.py \
#    --src_dir "../../train_log_v2/afhq_cat_val" \
#    --tgt_dir "../../train_log_v2/stylized_afhg_cat_val"

CUDA_VISIBLE_DEVICES=1 python lpips.py \
    --src_dir "../../train_log_v2/photo_scene_val" \
    --tgt_dir "../../train_log_v2/stylized_photo_scene_val"
