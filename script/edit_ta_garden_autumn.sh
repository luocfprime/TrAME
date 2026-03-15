python launch.py --config \
configs/edit-ta-garden.yaml \
--train \
--gpu \
0 \
system.guidance.prompt="an autumn garden" \
system.guidance.source_prompt="a summer garden" \
data.source=<scene_dir> \
system.gs_source=<path_to_point_cloud.ply>