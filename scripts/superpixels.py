import matplotlib.pyplot as plt
import numpy as np
import torch
from crf import pass_pseudomask_or_ground_truth, process_batch_crf
from PIL import Image
from skimage.segmentation import mark_boundaries, slic
from tifffile import imread
from tqdm import tqdm

from scripts.config import (
    DEVICE,
    N_SEGMENTS_RAPID,
    SLIC_COMPACTNESS,
    SUPERPIXEL_OVERLAP_THRESHOLD_RAPID,
)


def create_superpixel_mask(
    argmax_prediction_per_class,
    image,
    threshold=SUPERPIXEL_OVERLAP_THRESHOLD_RAPID,
    class_indx=1,
    N_SEGMENTS=N_SEGMENTS_RAPID,
    compactness=SLIC_COMPACTNESS,
    sigma=1,
    start_label=1,
    device=DEVICE,
):
    # get superpixels
    image = image.cpu().detach().numpy()
    all_superpixels_mask = torch.from_numpy(
        slic(
            image,
            n_segments=N_SEGMENTS,
            compactness=compactness,
            sigma=sigma,
            start_label=start_label,
        )
    )
    hadamard = all_superpixels_mask.to(device) * argmax_prediction_per_class.to(device)
    overlap = (hadamard / class_indx).type(torch.IntTensor)
    # Instantiate base mask
    base_mask = torch.zeros(overlap.shape)
    # Get numbers to list, start from second element because first is 0
    relevant_superpixels = torch.unique(overlap).int().tolist()[1:]
    for superpixel in relevant_superpixels:
        temp = overlap.clone()
        org = all_superpixels_mask.clone()
        #   # Check how many are non-zero in superpixel mask
        temp[temp != superpixel] = 0
        org[org != superpixel] = 0
        # Check how many are non-zero in overlap
        # Determine share of pixels
        share = torch.count_nonzero(temp).item() / torch.count_nonzero(org).item()
        # Add superpixel as ones to base mask if share is over threshold
        if share > threshold:
            # bring org values to one
            org = org / torch.unique(org)[1].item()
            base_mask += org
    # make values in base_mask equal the class value
    base_mask = base_mask * class_indx
    return base_mask.type(torch.IntTensor)


def visualize_superpixels(boundaries, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(25, 10))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        if i >= 0:  # and (i != 4 and i != 5):
            plt.title(" ".join(name.split("_")).title())
            plt.imshow(mark_boundaries(image, boundaries))
    plt.show()


def export_superpixel_crf_masks_for_dataset(
    dataset, export_path, device=DEVICE, save_as_png=True
):
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
        mask[mask > 0] = 1
        sp_mask = create_superpixel_mask(mask, img, N_SEGMENTS=200, threshold=0.60)
        img, sp_mask = img.to(device), sp_mask.to(device)
        pseudomask = process_batch_crf(img, sp_mask)
        pseudomask = pass_pseudomask_or_ground_truth(mask.to(device), pseudomask)
        if save_as_png == True:
            pseudomask = Image.fromarray(np.uint8(pseudomask.cpu().detach() * 255), "L")
            output_path_mask = (export_path + "/" + _.split("/")[-1]).replace(
                "tif", "png"
            )
            pseudomask.save(output_path_mask, quality=100, subsampling=0)
        else:
            output_path_mask = (export_path + "/" + _.split("/")[-1]).replace(
                "tif", "pt"
            )
            torch.save(pseudomask, output_path_mask)


def return_superpixel_crf_masks(dataset, device=DEVICE):
    images = dataset.X
    masks = dataset.Y
    mask_dict = {}
    for i, _ in tqdm(enumerate(images)):
        # load image
        img = torch.tensor(imread(_))
        # load mask
        if ".tif" in masks[i]:
            mask = torch.tensor(imread(masks[i])).long()
        elif ".png" in masks[i]:
            mask = torch.Tensor(np.array(Image.open(masks[i]))).long()
        # initialize base masks once
        if i == 0:
            base_masks = torch.zeros(
                [
                    len(images),
                    mask.shape[0],
                    mask.shape[1],
                ],
                dtype=mask.dtype,
                layout=mask.layout,
                device=mask.device,
            )
        mask[mask > 0] = 1
        sp_mask = create_superpixel_mask(mask, img, N_SEGMENTS=200, threshold=0.60)
        img, sp_mask = img.to(device), sp_mask.to(device)
        pseudomask = process_batch_crf(img, sp_mask)
        pseudomask = pass_pseudomask_or_ground_truth(mask.to(device), pseudomask)
        base_masks[i] += pseudomask
        mask_dict[i] = masks[i].replace("tif", "pt").replace("png", "pt").split("/")[-1]
    return base_masks, mask_dict
