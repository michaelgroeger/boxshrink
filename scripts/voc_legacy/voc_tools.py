import os
import random
import re
import shutil
import xml.etree.ElementTree as ET
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#############################################################################################################################
# This is a collection of scripts we wrote for our first experiments on PascalVOC.                                          #
# We later needed to switch the datasets because we were lacking the resources for a dataset of the size of PascalVOC.      #
# We still publish it here since some people might find them useful and we uses some of the scripts in the Boxshrink paper. #
#############################################################################################################################


def get_value_of_tag(filepath, tag):
    """Check for file if condition is fullfilled

    Args:
        filepath (string): path to file holding annotation data
        tag (string)): tag to look for
        condition (string, integer, float): Condition the tag should fullfill

    Returns:
        [string or None]: None if condition is not fullfilled, string if condition is met
    """
    # Reading the data inside the xml
    # file to a variable under the name
    # data
    with open(filepath, "r") as f:
        annotation_data = f.read()

    soup = BeautifulSoup(annotation_data, features="lxml")
    return str(soup.find(tag).string)


# Function to get only segmented, only unsegmented or all images -> Create dataset
def create_dataset(xml_files, tag, condition):
    """Creates a dataset based on a tag and a condition of that tag. Can be used to train on subsets of the
    PascalVOC dataset.

    Args:
        xml_files (list): list of path to files holding annotation data
        tag (string): tag to look for
        condition (string, integer, float): condition the tag should fullfill

    Returns:
        [list]: dataset holding the image paths
    """
    # create imageset
    return (
        [
            xml_files[file]
            for file in tqdm(range(len(xml_files)), desc="Creating dataset")
            if get_value_of_tag(xml_files[file], tag) == str(condition)
        ],
    )


# Function to get image & segmentation sets
def get_image_and_segmentation_sets(dataset, image_path, segmentation_path):
    """Takes as input path to xml annotation files. Will return two lists with path to
    Images and segmentation files.

    Args:
        dataset (list): List holding path information to xml annotations
        image_path (list): List holding path information to image data
        segmentation_path (list): List holding path information to segmentation data

    Returns:
        image_set: List with image paths
        segmentation_set: List with segmentation paths
    """
    image_set = []
    segmentation_set = []
    for index in tqdm(range(len(dataset)), desc="Building image and segmentation sets"):
        annotation = dataset[index][0]
        image_name = annotation.split("/")[-1].replace("xml", "jpg")
        segmentation_name = annotation.split("/")[-1].replace("xml", "png")
        image_set.append(image_path + "/" + image_name)
        segmentation_set.append(segmentation_path + "/" + segmentation_name)
    return image_set, segmentation_set


# Function to create validation, train, test splits
def train_test_val_split(
    image_set,
    segmentation_set,
    test_size=0.20,
    validation_size=0.10,
    seed=42,
    clean_validation=False,
    clean_segmentation_path=None,
):
    """Create train, test, validation splits of initial datasets.

    Args:
        image_set (list): List holding path information to images
        segmentation_set (list): List holding path information to segmentations
        test_size (Float): Float between 0.0 - 1.0 to determine test data size
        validation_size (Float): Float between 0.0 - 1.0 to determine the validation dataset size
        seed (int, optional): Set seed for sampling. Defaults to 42.
        clean_validation (bool, optional): Whether to use clean human made segmentation or pseudolabels. Defaults to False.
        clean_segmentation_path (String, optional): If you wish to use human labels then provide path. Defaults to None.

    Returns:
        [type]: [description]
    """
    # Turn percentage into concrete number of samples to draw
    random.seed(seed)
    validation_size = round(len(image_set) * validation_size)
    validation_indexes = random.sample(range(len(image_set)), validation_size)
    X_val = [image_set[i] for i in validation_indexes]
    y_val = [segmentation_set[i] for i in validation_indexes]

    # Drop images and segmentations separated for validation
    image_set = [image for image in image_set if image not in X_val]
    segmentation_set = [
        segmentation for segmentation in segmentation_set if segmentation not in y_val
    ]
    # Rebuild path to clean annotations
    if clean_validation == True:
        y_val = [
            clean_segmentation_path
            + "/"
            + re.split("/", segmentation)[-1].replace("jpg", "png")
            for segmentation in y_val
        ]
    X_train, X_test, y_train, y_test = train_test_split(
        image_set, segmentation_set, test_size=test_size, random_state=seed
    )
    return (X_train, y_train, X_test, y_test, X_val, y_val)


def export_dataset(dataset, output_path):
    """Export created dataset. Will be data: data: [path_to_img_1, ..., path_to_img_n]

    Args:
        dataset (list): list of paths
        output_path (string)
    """
    data = {"data": dataset}
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)


def import_dataset(input_path):
    """Import dataset create with export dataset function.

    Args:
        input_path (string)

    Returns:
        List
    """
    df = pd.read_csv(input_path)
    return df.values.tolist()


def check_box_coordinates(box_coordinate):
    """Catch case where box coordinate is of type float in PascalVOC xml annotation file and convert to int.

    Args:
        box_coordinate (string): Box coordinate field vale from xml file.

    Returns:
        int: Box coordinate as integer
    """
    if "." in box_coordinate:
        return int(float(box_coordinate))
    else:
        return int(box_coordinate)


def get_bounding_boxes(xml_file: str):
    """Returns bounding box information from PascalVOC xml annotation file

    Args:
        xml_file (str): path to file holding annotation information

    Returns:
        list: List with bounding box information
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter("object"):

        filename = root.find("filename").text

        ymin, xmin, ymax, xmax = None, None, None, None
        classname = str(boxes.find("name").text)
        ymin = check_box_coordinates(boxes.find("bndbox/ymin").text)
        xmin = check_box_coordinates(boxes.find("bndbox/xmin").text)
        ymax = check_box_coordinates(boxes.find("bndbox/ymax").text)
        xmax = check_box_coordinates(boxes.find("bndbox/xmax").text)
        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append({classname: list_with_single_boxes})

    return filename, list_with_all_boxes


def create_class_color_code_csv(
    dataset, output_path=None, background=False, return_dataframe=False
):
    """Creates a csv that has all unique categories in the given dataset and
    generates color codes for each of the categories based on random numbers.

    Args:
        dataset (list): list holding path information to annotation files
        output_path (string): path to save the csv to, please also provide the name of the csv e.g. path/to/csv.csv
        background (boolean): whether to include a background class or not
        return_dataframe (boolean): whether to return a dataframe or not
    """
    # Get categories
    categories = []
    for file in tqdm(range(len(dataset)), desc="Collecting Categories"):
        category = get_value_of_tag(dataset[file], "name")
        if category not in categories:
            categories.append(category)
    # Generate color codes
    # set seed to ensure reproducibility
    np.random.seed(0)
    red = [int(np.random.randint(0, 255, 1)) for category in categories]
    green = [int(np.random.randint(0, 255, 1)) for category in categories]
    blue = [int(np.random.randint(0, 255, 1)) for category in categories]
    # Create class indexes
    indexes = [index for index in range(len(categories))]
    # Check whether to add background class or not
    if background == True:
        categories.insert(0, "background")
        red.insert(0, 0)
        green.insert(0, 0)
        blue.insert(0, 0)
        indexes = [index + 1 for index in indexes]
        indexes.insert(0, 0)
    # Build dataframe
    data = {
        "index": indexes,
        "category": categories,
        "red": red,
        "green": green,
        "blue": blue,
    }
    df = pd.DataFrame(data)
    if return_dataframe == False:
        df.to_csv(output_path, index=False)
    else:
        return df


# Function to generate masks from bounding boxes
def generate_mask_from_box(
    dataset,
    category_data,
    output_path,
):
    """Generate weak segmentation masks from dataset.

    Args:
        dataset (List): holding paths to images
        category_data (str): Path to file holding class to color mapping
        output_path (str): where to save the masks
    """
    category_data = pd.read_csv(category_data)
    for element in tqdm(range(len(dataset)), desc="Creating Masks"):
        image_width, image_height = (
            int(get_value_of_tag(dataset[element][0], "width")),
            int(get_value_of_tag(dataset[element][0], "height")),
        )
        channels = 3
        # Create mask
        mask = np.zeros([image_height, image_width, channels], dtype=np.uint8)
        # Get all boxes present on the image
        boxes = get_bounding_boxes(dataset[element][0])
        for box in boxes[1]:
            # Get category
            category = list(box.keys())[0]
            # Get color code
            red, green, blue = (
                np.uint8(category_data[category_data["category"] == category]["red"]),
                np.uint8(category_data[category_data["category"] == category]["green"]),
                np.uint8(category_data[category_data["category"] == category]["blue"]),
            )
            # get coordinates
            ymin, xmin, ymax, xmax = (
                box[category][0],
                box[category][1],
                box[category][2],
                box[category][3],
            )
            if ymax != image_height:
                ymax = ymax + 1
            if xmax != image_width:
                xmax = xmax + 1
            mask[xmin:xmax, ymin:ymax] = [red[0], green[0], blue[0]]
        mask = Image.fromarray(mask, mode="RGB")
        output_path_mask = (
            output_path + "/" + get_value_of_tag(dataset[element][0], "filename")
        ).replace("jpg", "png")
        mask.save(output_path_mask, quality=100, subsampling=0)


def visualize_mask(image_path, path_to_masks):
    """Will return an overlay of the image and its mask

    Args:
        image_path (string): path to image you want to visualize
        path_to_masks (string): path to masks
    """
    # Ground truth image
    background = Image.open(image_path)
    # Generate path to mask
    image_path = re.split(r"/", image_path)
    image_name = image_path[-1]
    mask_name = image_name.replace("jpg", "png")
    mask_path = path_to_masks + "/" + mask_name
    # load mask
    overlay = Image.open(mask_path)
    # Ensure same encoding
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")
    # Create new image from overlap and make overlay 50 % transparent
    new_img = Image.blend(background, overlay, 0.5)
    new_img.show()


def resize_images_in_folder(
    path_to_folder, path_to_output_folder, new_x_res, new_y_res
):
    """

    Args:
        path_to_folder (string): Folder holding original images
        path_to_output_folder (string): where to save resized images
        new_x_res (int): New max x resolution
        new_y_res (int): New max y resolution
    """
    files = [f for f in os.listdir(path_to_folder) if isfile(join(path_to_folder, f))]
    for file in tqdm(range(len(files)), desc="Rescaling images: "):
        # open image
        img = Image.open(join(path_to_folder, files[file]))
        # resize image
        img = img.resize((new_x_res, new_y_res), resample=Image.NEAREST)
        # save image in outputpath
        img.save(join(path_to_output_folder, files[file]), quality=100, subsampling=0)


def get_smallest_imagesize_in_folder(path_to_folder):
    """Returns smallest side of images in folder.

    Args:
        path_to_folder (string): path to folder

    Returns:
        int: shortest side of images in folder
    """
    files = [f for f in os.listdir(path_to_folder) if isfile(join(path_to_folder, f))]
    sizes = np.array([])
    for file in tqdm(range(len(files)), desc="Getting smallest side: "):
        # open image
        size_image = Image.open(join(path_to_folder, files[file])).size
        # Convert image size to numpy array
        sizes = np.append(sizes, size_image[0], size_image[1])
    # Return smallest value
    return int(sizes.min())


# All delivered segmentation masks have a white boundary which one might want to eliminate
# the color code is [224, 224, 192]
def drop_color_in_image(
    path,
    output_path,
    color_code=[224, 224, 192],
):
    """sets pixels of a certain value to zero

    Args:
        path (string): path to image
        output_path (string): where to save altered image
        color_code (list, optional): RGB value to be set to zero. Defaults to [224, 224, 192].
    """
    # open image
    img = Image.open(path)
    # Convert to RGB
    img = img.convert("RGB")
    # Convert to numpy
    img_np = np.array(img)
    # Change color code to black
    img_np[img_np[:, :, :] == color_code] = 0
    # Generate image name
    img_name = re.split("/", path)[-1]
    # Export image
    new_img = Image.fromarray(img_np)
    path = os.path.join(output_path, img_name)
    new_img.save(path, quality=100, subsampling=0)


def export_visualize(path, **images):
    """Plots image e.g. img, img+ground truth, img+prediction in one line
    and exports it.

    Args:
        path (string): where to save the image
    """
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plot = plt.gcf()
    plot.savefig(path)


def IoU(image, gt, model, eps=0.0000000001, mode="mean"):
    """Calculation of IoU metric. The function checks if we have a batchsize of 1
    then the IoU of the single image with the ground truth mask is returned. If the batchsize is
    bigger then the IoU over the batch is calculated.

    Args:
        gt (tensor): ground truth class mask
        pred (tensor): prediction made by the model.
        eps (float, optional): Epsilon to avoid zero division error. Defaults to 0.0000000001.
        mode (string, optional): Whether to return the entire tensor or the mean. Defaults to mean. Can be 'mean'
        or 'tensor'

    Returns:
        [float]: IoU score
    """
    batchsize = gt.shape[0]
    ious = []
    with torch.no_grad():
        if gt.shape[0] > 1:
            pass
            # Unfold the images and calculate prediction for each of them
            for i in range(batchsize):
                current_gt = gt[i, :, :]
                current_pred = torch.argmax(
                    model(image[i, :, :].unsqueeze(0)), dim=1, keepdim=True
                )[:, -1, :, :]
                intersection = torch.logical_and(current_gt, current_pred) + eps
                union = torch.logical_or(current_gt, current_pred) + eps
                iou_score = torch.sum(intersection) / torch.sum(union)
                ious.append(iou_score.item())
            if mode == "mean":
                return torch.tensor(ious, dtype=torch.float64).mean()
            if mode == "tensor":
                return torch.tensor(ious, dtype=torch.float64)
            else:
                return None
        else:
            # Calculate IoU for batchsize = 1
            pred = torch.argmax(model(image), dim=1, keepdim=True)[:, -1, :, :]
            intersection = torch.logical_and(gt, pred) + eps
            union = torch.logical_or(gt, pred) + eps
            iou_score = torch.sum(intersection) / torch.sum(union)
            return iou_score


# What directories to expect
def copy_images_to_directories(root, directories, datasets, select_subset_of_images):
    """Copy images from datasets into new directory to keep the original data separated from any
    data with which experiments were done. Also to separate images into the representative dataset folder.

    Args:
        root (string): Root directory where the new datasets folders should be created in
        directories (list): List of names for the new directories. e.g. ["X_train", "y_train", ...]
        datasets ([type]): [description]
        select_subset_of_images ([type]): [description]
    """
    select_subset_of_images = select_subset_of_images
    for index in range(len(directories)):
        # Build new path where to copy the images to
        new_path = root + "/" + directories[index]
        # If the new path has already images from earlier experiments, delete those
        if os.path.exists(new_path):
            shutil.rmtree(new_path)
        # create folder to save new copied images
        os.mkdir(new_path)
        # go through each dataset
        dataset = datasets[index]
        # Scenario where we only want some images to be copied and not the entire dataset
        if select_subset_of_images == 0:
            for i in tqdm(
                range(len(datasets[index])), desc=f"Copy images to {directories[index]}"
            ):
                new_image_path = new_path + "/" + dataset[i].split("/")[-1]
                shutil.copy(dataset[i], new_image_path)
        else:
            # Scenario where we want to work with the entire dataset
            for i in tqdm(
                range(select_subset_of_images),
                desc=f"Copy images to {directories[index]}",
            ):
                new_image_path = new_path + "/" + dataset[i].split("/")[-1]
                shutil.copy(dataset[i], new_image_path)
