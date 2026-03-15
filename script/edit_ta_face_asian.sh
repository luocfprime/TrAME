python launch.py --config \
configs/edit-ta-face.yaml \
--train \
--gpu \
0 \
system.guidance.prompt="an Asian man" \
system.guidance.source_prompt="a white man" \
data.source=<scene_dir> \
system.gs_source=<path_to_point_cloud.ply> \
system.seg_prompt="a person" \
data.init_inference_step=0
