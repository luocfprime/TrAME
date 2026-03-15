python launch.py \
--config \
configs/edit-ta-fangzhou.yaml \
--train \
--gpu \
0 \
system.guidance.prompt="a Hulk" \
system.guidance.source_prompt="a face of a man" \
data.source=<scene_dir> \
system.gs_source=<path_to_point_cloud.ply> \
system.seg_prompt="a person" \
data.init_inference_step=0