python launch.py --config \
configs/edit-ta-face.yaml \
--train \
--gpu \
0 \
system.guidance.prompt="An elderly man with a small, neatly trimmed mustache, dressed in a jacket of yellowish khaki, his face flushed red." \
system.guidance.source_prompt="a face of a person" \
data.source=<scene_dir> \
system.gs_source=<path_to_point_cloud.ply> \
system.loggers.wandb.enable=false \
system.seg_prompt="a person" \
system.guidance.self_kv_edit_step_thresh=0.0 \
data.init_inference_step=5
