import cv2
from torchmetrics import JaccardIndex
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import numpy as np
import torch 
from tqdm import tqdm
from PIL import Image
from config import CLASSES, IOU_THRESHOLD, MASK_OCCUPANCY_THRESHOLD
from tifffile import imread

jaccard_crf = JaccardIndex(num_classes=len(CLASSES), average='None', ignore_index=0)

def crf(img_org, mask, pb_sxy=(25,25), pb_srgb=(10,10,10), pg_sxy=(5,5)):
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

    gauss_img = cv2.GaussianBlur(img_np, (31, 31), 0)

    bilat_img = cv2.bilateralFilter(img_np, d=10, sigmaColor=80, sigmaSpace=80)

    n_classes = 2
    feat_first = im_softmax.transpose((2, 0, 1)).reshape((n_classes,-1))
    unary = unary_from_softmax(feat_first)
    unary = np.ascontiguousarray(unary)
    img_np = img_np.copy(order='C')

    d = dcrf.DenseCRF2D(img_np.shape[1], img_np.shape[0], n_classes)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=pg_sxy, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    d.addPairwiseBilateral(sxy=pb_sxy, srgb=pb_srgb, rgbim=img_np, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(10)
    res = np.argmax(Q, axis=0).reshape((img_np.shape[0], img_np.shape[1]))
    # mutliply by one because this is the class index
    res *= 1
    return torch.from_numpy(res).type(torch.IntTensor)

def process_batch_crf(img_org, mask):
    if img_org.dim() > 3:
        batch = torch.zeros(mask.shape, dtype=torch.int64)
        for i in range(len(img_org)):
            pseudomask = crf(img_org[i], mask[i])
            batch[i,:,:] = pseudomask
        return batch
    else:
        return crf(img_org, mask)
    

def pass_pseudomask_or_ground_truth(masks, pseudomasks, iou_threshold=IOU_THRESHOLD, mask_occupancy_threshold=MASK_OCCUPANCY_THRESHOLD, IoU=jaccard_crf):
    if masks.dim() > 2:
        batch = torch.zeros(masks.shape, dtype=torch.float32)
        pseudomasks_count = 0
        for i in range(masks.shape[0]):
            total_mask_occupancy = torch.count_nonzero(masks[i]) / (masks[i].shape[0] * masks[i].shape[1])
            if IoU(pseudomasks[i].unsqueeze(0), masks[i].unsqueeze(0)) < iou_threshold or total_mask_occupancy < mask_occupancy_threshold:
                batch[i] = masks[i]
            else: 
                batch[i] += pseudomasks[i]
                pseudomasks_count += 1
        print(f"Of {masks.shape[0]} masks {pseudomasks_count} were used")
        return batch
    else:
        total_mask_occupancy = torch.count_nonzero(masks) / (masks.shape[0] * masks.shape[1])
        if IoU(pseudomasks.unsqueeze(0), masks.unsqueeze(0)) < iou_threshold or total_mask_occupancy < mask_occupancy_threshold:
            return masks
        else:
            return pseudomasks

def export_crf_masks_for_train_data(dataset, export_path):
        images = dataset.X
        masks = dataset.Y
        for i, _ in tqdm(enumerate(images)):
            # load image
            img = torch.tensor(imread(_))
            # load mask
            if ".tif" in masks[i]:
                mask = torch.tensor(imread(masks[i])).long()
            elif ".png" in masks[i]:
                mask = torch.Tensor(np.array(Image.open(masks[i]))).long()
            mask[mask>0] = 1    
            img, mask = img, mask
            pseudomask = process_batch_crf(img, mask)
            pseudomask = pass_pseudomask_or_ground_truth(mask, pseudomask)
            pseudomask = Image.fromarray(np.uint8(pseudomask.cpu().detach() * 255) , 'L')
            output_path_mask = (
            export_path + "/" + _.split('/')[-1]
            ).replace("tif", "png")
            pseudomask.save(output_path_mask, quality=100, subsampling=0)