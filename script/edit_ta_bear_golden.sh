python launch.py --config \
configs/edit-ta-bear.yaml \
--train \
--gpu \
0 \
system.guidance.prompt="a shiny bear statue made of pure gold" \
system.guidance.source_prompt="a stone bear statue" \
data.source=<scene_dir> \
system.gs_source=<path_to_point_cloud.ply> \
system.loggers.wandb.enable=false \
system.loggers.wandb.name="edit_ta_bear_gold" \
system.seg_prompt="bear"