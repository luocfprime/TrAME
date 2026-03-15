python launch.py --config \
configs/edit-ta-garden.yaml \
--train \
--gpu \
0 \
system.guidance.prompt="a blue-and-white porcelain vase on a wooden table" \
system.guidance.source_prompt="a wooden vase on a wooden table" \
data.source=<scene_dir> \
system.gs_source=<path_to_point_cloud.ply> \
system.loggers.wandb.enable=false \
system.loggers.wandb.name="edit_ta_garden_vase" \
system.seg_prompt="a vase"