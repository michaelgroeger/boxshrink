import os

import torch
from superpixels import return_superpixel_crf_masks

from scripts.config import DATA_DIR
from scripts.dataset import Colonoscopy_Dataset
from scripts.tools import human_sort, return_files_in_directory


def rapid_boxshrink():
    image_files = return_files_in_directory(DATA_DIR + "/original", ".tif")
    box_files = return_files_in_directory(DATA_DIR + "/boxmasks", ".png")
    # Ensure files are in correct order
    human_sort(image_files)
    human_sort(box_files)
    dataset = Colonoscopy_Dataset(image_files[:5], box_files[:5])

    return return_superpixel_crf_masks(dataset)


def load_test_masks_in_one_tensor(base_masks, files, mask_dict):
    for i, _ in enumerate(files):
        mask_number = mask_dict[i]
        mask = [m for m in files if mask_number in m]
        print(f"Load mask {mask} at position {i}")
        base_masks[i] = torch.load(mask[0])
    return base_masks


def test_rapid_boxshrink():
    TESTING_DIR = os.path.join(os.getcwd(), "tests/test_rapid_boxshrink_masks")
    generated_masks, mask_dict = rapid_boxshrink()
    test_mask_files = return_files_in_directory(TESTING_DIR, ".pt")
    print(mask_dict)
    test_masks = torch.zeros_like(generated_masks)
    test_masks = load_test_masks_in_one_tensor(test_masks, test_mask_files, mask_dict)
    assert torch.all(
        test_masks.eq(generated_masks)
    ), f"Generated embedding masks and test masks not equal"
