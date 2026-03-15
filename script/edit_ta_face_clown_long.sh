python launch.py --config \
configs/edit-ta-face.yaml \
--train \
--gpu \
0 \
system.guidance.prompt="A clown with a round, white-painted face. His bulbous red nose sits at the center, and thick black eyeliner emphasizes his wide eyes. A puffy flaming red wig surrounds his head like a cloud. He is smiling a wide grin." \
system.guidance.source_prompt="a face of a person" \
system.guidance.negative_prompt="low resolution, defects, over-saturate, unnatural" \
data.source=<scene_dir> \
system.gs_source=<path_to_point_cloud.ply> \
system.seg_prompt="a person" \
data.init_inference_step=5

#system.anchor_weight_init=0.1 \
#system.anchor_weight_multiplier=1.3 \
#system.gs_lr_scaler=5 \
#system.gs_final_lr_scaler=5 \
#system.color_lr_scaler=5 \
#system.opacity_lr_scaler=2 \
#system.scaling_lr_scaler=2 \
#system.rotation_lr_scaler=2 \

#system.loss.lambda_anchor_color=0 \
#system.loss.lambda_anchor_geo=50 \
#system.loss.lambda_anchor_scale=50 \
#system.loss.lambda_anchor_opacity=50 \
#system.densify_from_iter=100 \
#system.densify_until_iter=1501 \
#system.densification_interval=100 \