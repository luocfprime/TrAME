python launch.py --config \
configs/edit-ta-bike.yaml \
--train \
--gpu \
0 \
system.guidance.prompt="an autumn park" \
system.guidance.source_prompt="a summer park" \
data.source=<scene_dir> \
system.gs_source=<path_to_point_cloud.ply> \
system.masked_loss=false \
system.loggers.wandb.enable=false \
system.loggers.wandb.name="edit_ta_bike_autumn"
