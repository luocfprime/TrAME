python launch.py --config \
configs/edit-ta-face.yaml \
--train \
--gpu \
0 \
system.guidance.prompt="a face of Albert Einstein" \
system.guidance.source_prompt="a face of a person" \
data.source=<scene_dir> \
system.gs_source=<path_to_point_cloud.ply> \
system.loggers.wandb.enable=false \
system.seg_prompt="a person" \
system.guidance.self_kv_edit_step_thresh=0.7 \
data.init_inference_step=10
