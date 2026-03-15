python launch.py --config \
configs/edit-ta-face.yaml \
--train \
--gpu \
0 \
system.guidance.prompt="a spider man in combat suit with face covered entirely by mask" \
system.guidance.source_prompt="a face of a person" \
system.max_densify_percent=0.01 \
system.anchor_weight_init_g0=0.05 \
system.anchor_weight_init=0.1 \
system.anchor_weight_multiplier=1.3 \
data.source=<scene_dir> \
system.gs_source=<path_to_point_cloud.ply> \
system.loggers.wandb.enable=false \
system.loggers.wandb.name="edit_ta_face_spider_man" \
system.seg_prompt="a person" \
system.guidance.self_kv_edit_step_thresh=0.0 \
data.init_inference_step=0

#system.loss.lambda_anchor_color=0 \
#system.loss.lambda_anchor_geo=50 \
#system.loss.lambda_anchor_scale=50 \
#system.loss.lambda_anchor_opacity=50 \
#system.densify_from_iter=100 \
#system.densify_until_iter=1501 \
#system.densification_interval=100 \