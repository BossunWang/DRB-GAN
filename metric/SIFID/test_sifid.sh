#CUDA_VISIBLE_DEVICES=1 python sifid_score.py \
#    --path2real "../../train_log_v2/style_ref_val" \
#    --path2fake "../../train_log_v2/stylized_afhg_cat_val" \
#    --images_suffix "png" \
#    --gpu 1

CUDA_VISIBLE_DEVICES=1 python sifid_score.py \
    --path2real "../../train_log_v2/style_ref_photo_scene_val" \
    --path2fake "../../train_log_v2/stylized_photo_scene_val" \
    --images_suffix "png" \
    --gpu 1
