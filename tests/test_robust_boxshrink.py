import os

import torch
import torchvision
from sklearn.model_selection import train_test_split

from scripts.config import DATA_DIR, DEVICE
from scripts.dataset import Colonoscopy_Dataset
from scripts.embeddings import (
    ResnetFeatureExtractor,
    return_embedding_masks_for_dataset,
)
from scripts.tools import human_sort, return_files_in_directory


def robust_boxshrink(data_dir, embedding_idr):
    image_files = return_files_in_directory(data_dir + "/original", ".tif")
    box_files = return_files_in_directory(data_dir + "/boxmasks", ".png")
    # Ensure files are in correct order
    human_sort(image_files)
    human_sort(box_files)
    X_train, _, y_train, _ = train_test_split(
        image_files, box_files, test_size=0.1, random_state=1
    )
    X_train, _, y_train, _ = train_test_split(
        X_train, y_train, test_size=0.11111, random_state=1
    )  # 0.1111 x 0.9 = 0.1

    dataset = Colonoscopy_Dataset(X_train[:5], y_train[:5])
    mean_f = torch.mean(
        torch.load(embedding_idr + "foreground_embedding.pt"),
        dim=0,
    )
    mean_b = torch.mean(
        torch.load(embedding_idr + "background_embedding.pt"),
        dim=0,
    )
    resnet = torchvision.models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V2")
    resnet.eval()
    feature_extract_model = ResnetFeatureExtractor(resnet)
    feature_extract_model.to(DEVICE)
    return return_embedding_masks_for_dataset(
        dataset, feature_extract_model, mean_f, mean_b
    )


def load_test_masks_in_one_tensor(base_masks, files, mask_dict):
    for i, _ in enumerate(files):
        mask_number = mask_dict[i]
        mask = [m for m in files if mask_number in m]
        base_masks[i] = torch.load(mask[0])
    return base_masks


def test_robust_boxshrink():
    TESTING_DIR = os.path.join(os.getcwd(), "tests/test_robust_boxshrink_masks")
    EMBEDDING_DIR = os.path.join(DATA_DIR, "mean_embeddings/")
    generated_masks, mask_dict = robust_boxshrink(DATA_DIR, EMBEDDING_DIR)
    test_mask_files = return_files_in_directory(TESTING_DIR, ".pt")
    test_masks = torch.zeros_like(generated_masks)
    test_masks = load_test_masks_in_one_tensor(test_masks, test_mask_files, mask_dict)
    assert torch.all(
        test_masks.eq(generated_masks)
    ), f"Generated embedding masks and test masks not equal"
