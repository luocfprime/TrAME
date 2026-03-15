python launch.py --config \
configs/edit-ta-face.yaml \
--train \
--gpu \
0 \
system.guidance.prompt="a face of an elf" \
system.guidance.source_prompt="a face of a person" \
data.source=<scene_dir> \
system.gs_source=<path_to_point_cloud.ply> \
system.loggers.wandb.enable=false \
system.loggers.wandb.name="edit_ta_face_elf" \
system.seg_prompt="a person" \
data.init_inference_step=0