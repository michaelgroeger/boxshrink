#############################################################################################################################
#                               Functions that are needed for the F-CRF functionality                                       #
#############################################################################################################################

import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
import torch
from pydensecrf.utils import unary_from_softmax
from torchmetrics import JaccardIndex

from scripts.config import (
    CLASSES,
    DEVICE,
    IOU_THRESHOLD,
    MASK_OCCUPANCY_THRESHOLD,
    NUM_INFERENCE,
    PAIRWISE_BILATERAL,
    PAIRWISE_GAUSSIAN,
    RGB_STD,
)

jaccard_crf = JaccardIndex(num_classes=len(CLASSES), average="none", ignore_index=0).to(
    DEVICE
)


def crf(
    img_org,
    mask,
    pb_sxy=PAIRWISE_BILATERAL,
    pb_srgb=RGB_STD,
    pg_sxy=PAIRWISE_GAUSSIAN,
    num_classes=len(CLASSES),
):
    img_np = img_org.cpu().detach().numpy()
    # Skip background class
    mask_np = mask.cpu().detach().numpy() * 255
    mask_np = mask_np.astype(np.uint8)
    mask_np[mask_np > 0] = 145
    img_np = img_np.astype(np.uint8)

    not_mask = np.invert(mask_np)
    not_mask = np.expand_dims(not_mask, axis=2)
    mask_np_processed = np.expand_dims(mask_np, axis=2)

    im_softmax = np.concatenate([not_mask, mask_np_processed], axis=2)
    im_softmax = im_softmax / 255.0
    n_classes = num_classes
    feat_first = im_softmax.transpose((2, 0, 1)).reshape((n_classes, -1))
    unary = unary_from_softmax(feat_first)
    unary = np.ascontiguousarray(unary)
    img_np = img_np.copy(order="C")

    d = dcrf.DenseCRF2D(img_np.shape[1], img_np.shape[0], n_classes)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(
        sxy=pg_sxy,
        compat=10,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    d.addPairwiseBilateral(
        sxy=pb_sxy,
        srgb=pb_srgb,
        rgbim=img_np,
        compat=10,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )
    Q = d.inference(NUM_INFERENCE)
    res = np.argmax(Q, axis=0).reshape((img_np.shape[0], img_np.shape[1]))
    # mutliply by one because this is the class index
    res *= 1
    return torch.from_numpy(res).type(torch.IntTensor)


def process_batch_crf(img_org, mask, device=DEVICE):
    if img_org.dim() > 3:
        batch = torch.zeros(mask.shape, dtype=torch.int64).to(device)
        for i in range(len(img_org)):
            pseudomask = crf(img_org[i], mask[i]).to(device)
            batch[i, :, :] = pseudomask
        return batch
    else:
        return crf(img_org, mask).to(device)


def pass_pseudomask_or_ground_truth(
    masks,
    pseudomasks,
    iou_threshold=IOU_THRESHOLD,
    mask_occupancy_threshold=MASK_OCCUPANCY_THRESHOLD,
    device=DEVICE,
    IoU=jaccard_crf,
):
    if masks.dim() > 2:
        batch = torch.zeros(masks.shape, dtype=torch.float32).to(device)
        pseudomasks_count = 0
        for i in range(masks.shape[0]):
            total_mask_occupancy = torch.count_nonzero(masks[i]) / (
                masks[i].shape[0] * masks[i].shape[1]
            )
            if (
                IoU(pseudomasks[i].unsqueeze(0), masks[i].unsqueeze(0)) < iou_threshold
                or total_mask_occupancy < mask_occupancy_threshold
            ):
                batch[i] = masks[i]
            else:
                batch[i] += pseudomasks[i]
                pseudomasks_count += 1
        return batch
    else:
        total_mask_occupancy = torch.count_nonzero(masks) / (
            masks.shape[0] * masks.shape[1]
        )
        if (
            IoU(pseudomasks.unsqueeze(0), masks.unsqueeze(0)) < iou_threshold
            or total_mask_occupancy < mask_occupancy_threshold
        ):
            return masks
        else:
            return pseudomasks
