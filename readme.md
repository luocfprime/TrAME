<h2 align="center">TrAME</h2>

<p align="center">
  Official implementation of<br>
  <strong>TrAME: Trajectory-Anchored Multi-View Editing for Text-Guided 3D Gaussian Manipulation</strong>
</p>

<p align="center">
  <a href="https://luocfprime.github.io/TrAME/">Project Page</a>
</p>

## Overview

This repository contains the official implementation of TrAME, a 3D Gaussian Splatting editing system built on top of GaussianEditor.

## Installation

Tested on Ubuntu with CUDA 11.8 and Python 3.8.

```bash
git clone <your-fork-or-this-repo-url>
cd TrAME
conda env create -f environment.yaml
mkdir -p extern
git clone git@github.com:luocfprime/infedit-unofficial.git extern/infedit-unofficial
pip install -e extern/infedit-unofficial
```

## Data Preparation

TrAME expects a pretrained Gaussian point cloud (`.ply`) and a COLMAP reconstruction directory.

```bash
bash download.sh
```

If you use your own scene, prepare a COLMAP folder and a trained Gaussian point cloud such as `point_cloud/iteration_7000/point_cloud.ply`.

## Demo

The training / export entry point is `launch.py`. TrAME-specific editing configs are under `configs/edit-ta*.yaml`.

A typical run looks like:

```bash
python launch.py \
  --config configs/edit-ta-face.yaml \
  --train \
  --gpu 0 \
  data.source=<scene_dir> \
  system.gs_source=<path_to_point_cloud.ply> \
  system.guidance.prompt="a face of Albert Einstein" \
  system.guidance.source_prompt="a face of a person" \
  system.seg_prompt="a person" \
  system.loggers.wandb.enable=false
```

Useful configs:

- `configs/edit-ta-face.yaml`
- `configs/edit-ta-person.yaml`
- `configs/edit-ta-fangzhou.yaml`
- `configs/edit-ta-garden.yaml`
- `configs/edit-ta-bike.yaml`
- `configs/edit-ta-face-ablation-wo-TAS.yaml`
- `configs/edit-ta-face-ablation-wo-TAS-wo-VCAC.yaml`

The `script/` directory contains more concrete examples for different scenes and prompts.

## Citation

```bibtex
@ARTICLE{10964679,
  author={Luo, Chaofan and Di, Donglin and Yang, Xun and Ma, Yongjia and Xue, Zhou and Chen, Wei and Gou, Xiaofei and Liu, Yebin},
  journal={IEEE Transactions on Multimedia},
  title={TrAME: Trajectory-Anchored Multi-View Editing for Text-Guided 3D Gaussian Manipulation},
  year={2025},
  volume={27},
  number={},
  pages={2886-2898},
  keywords={Three-dimensional displays;Image reconstruction;Noise;Noise reduction;Semantics;Optimization;Diffusion models;Schedules;Iterative methods;Text to image;Attention mechanism;diffusion models;3D scene editing;3D Gaussian splatting},
  doi={10.1109/TMM.2025.3557618}
}
```

## Acknowledgement

This repository is built upon GaussianEditor and also depends on several open-source projects:

- [GaussianEditor](https://github.com/buaacyw/GaussianEditor)
- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Threestudio](https://github.com/threestudio-project/threestudio)
- [InfEdit Unofficial](https://github.com/luocfprime/infedit-unofficial)
