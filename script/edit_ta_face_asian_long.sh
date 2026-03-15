python launch.py --config \
configs/edit-ta-face.yaml \
--train \
--gpu \
0 \
system.guidance.prompt="A middle-aged Asian man with neatly combed black hair, wearing thin rectangular glasses perched on his nose. He has a calm expression, dressed in a dark blue jacket." \
system.guidance.source_prompt="a white man" \
data.source=<scene_dir> \
system.gs_source=<path_to_point_cloud.ply> \
system.seg_prompt="a person" \
data.init_inference_step=0
