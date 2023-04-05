# BoxShrink: From Bounding Boxes to Segmentation Masks

This is the repository for the corresponding MICCAI-MILLanD Workshop 2022 paper [BoxShrink: From Bounding Boxes to Segmentation Masks](https://arxiv.org/abs/2208.03142)

## Overview

We present two algorithms how one can process bounding boxes to pseudo segmentation masks in a binary class segmenetation setting:

1. rapidBoxshrink: Works by a simple thresholding and overlapping strategy between the initial bounding box and
the generated superpixels. This algorithm will reject superpixels that don't overlap to a certain percentage with the bounding box and then run a F-CRF on the pseudomask. ![rapidBoxshrink-Overview](/images/rapidBoxshrink_overview.png)

2. robustBoxshrink: Compares the superpixels on the boundary with the mean foreground and background embedding of the training dataset. Those whose cosine distance is closer to the background embedding are being rejected. Finally, a F-CRF is being run on the pseudomask. ![robustBoxshrink-Overview](/images/robustBoxshrink_Overview.png)

## Installation 

Please follow these steps to install the environment.

```zsh
# Creates new conda environment
conda create -n boxshrink python=3.10.8 ipython
# Activates conda environment
conda activate boxshrink
# Makes script folder callable as module in python scripts
conda install conda-build
conda develop ./scripts
# Installs dependencies
pip install -r requirements.txt
# Make conda available in jupyter notebook
conda install ipykernel
python -m ipykernel install --user --name=boxshrink
```

## Usage

Please check the config file in `scripts/config` to set paths and hyperparameters. Please have a look at the notebook files if you want to generate bounding boxes from masks, run rapidBoxshrink or robustBoxshrink. After you generated the masks feel free to use them as training input as shown in `train.ipynb`. Have fun!

## Citation

If you use this work please cite:

```latex
@inproceedings{groger2022boxshrink,
  title={BoxShrink: From Bounding Boxes to Segmentation Masks},
  author={Gröger, Michael and Borisov, Vadim and Kasneci, Gjergji},
  booktitle={Workshop on Medical Image Learning with Limited and Noisy Data},
  pages={65--75},
  year={2022},
  organization={Springer}
}
```
