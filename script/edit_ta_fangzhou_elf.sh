python launch.py \
--config \
configs/edit-ta-fangzhou.yaml \
--train \
--gpu \
0 \
system.guidance.prompt="an elf" \
system.guidance.source_prompt="a person" \
data.source=<scene_dir> \
system.gs_source=<path_to_point_cloud.ply> \
system.loggers.wandb.enable=false \
system.loggers.wandb.name="edit_ta_fangzhou_elf" \
system.seg_prompt="a person" \
system.guidance.self_kv_edit_step_thresh=0.0 \
data.init_inference_step=8