#CUDA_VISIBLE_DEVICES=1 python score_infinity.py \
#    --path "../../train_log_v2/style_ref_photo_scene_val" \
#    --out_path "stylized_afhg_output_statistics.npz"

CUDA_VISIBLE_DEVICES=1 python score_infinity.py \
    --path "../../train_log_v2/stylized_photo_scene_val" \
    --out_path "stylized_photo_scene_output_statistics.npz"
