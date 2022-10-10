# BoxShrink: From Bounding Boxes to Segmentation Masks
This is the repository for the corresponding MICCAI-MILLanD Workshop 2022 paper [BoxShrink: From Bounding Boxes to Segmentation Masks](https://arxiv.org/abs/2208.03142)

## Introduction
We present two algorithms how one can process bounding boxes to pseudo segmentation masks in a binary class segmenetation setting:
1. rapidBoxshrink: Works by a simple thresholding and overlapping strategy between the initial bounding box and
the generated superpixels. This algorithm will reject superpixels that don't overlap to a certain percentage with the bounding box and then run a F-CRF on the pseudomask. ![rapidBoxshrink-Overview](/images/rapidBoxshrink_overview.png)
2. robustBoxshrink: Compares the superpixels on the boundary with the mean foreground and background embedding of the training dataset. Those whose cosine distance is closer to the background embedding are being rejected. Finally, a F-CRF is being run on the pseudomask. ![robustBoxshrink-Overview](/images/robustBoxshrink_Overview.png)

## Usage
Please check the config file in `scripts/config` to set paths and hyperparameters. Please have a look at the notebook files if you want to generate bounding boxes from masks, run rapidBoxshrink or robustBoxshrink. After you generated the masks feel free to use them as training input as shown in `train.ipynb`. Have fun!

## Citation
If you use this work please cite:
```
@inproceedings{groger2022boxshrink,
  title={BoxShrink: From Bounding Boxes to Segmentation Masks},
  author={Gröger, Michael and Borisov, Vadim and Kasneci, Gjergji},
  booktitle={Workshop on Medical Image Learning with Limited and Noisy Data},
  pages={65--75},
  year={2022},
  organization={Springer}
}
```
