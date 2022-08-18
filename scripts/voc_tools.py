from bs4 import BeautifulSoup
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import re
from sklearn.model_selection import train_test_split
from os.path import isfile, join
import torch
import random
import shutil
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

#############################################################################################################################
# This is a collection of scripts we wrote for our first experiments on PascalVOC.                                          #
# We later needed to switch the datasets because we were lacking the resources for a dataset of the size of PascalVOC.      # 
# We still publish it here since some people might find them useful and we uses some of the scripts in the Boxshrink paper. #
#############################################################################################################################
label_colors = np.array(
    [
        (0, 0, 0),  # Class 0
        (192, 128, 128),  # Class  1
        (192, 0, 0),  # Class  2
        (0, 64, 128),  # Class  3
        (64, 0, 0),  # Class  4
        (128, 64, 0),  # Class 5
        (0, 192, 0),  # Class   6
        (128, 128, 0),  # Class  7
        (128, 128, 128),  # Class 8
        (0, 128, 0),  # Class  9
        (0, 0, 128),  # Class   10
        (128, 192, 0),  # Class   11
        (64, 128, 0),  # Class 12
        (192, 0, 128),  # Class  13
        (64, 0, 128),  # Class   14
        (128, 0, 0),  # Class  15
        (0, 128, 128),  # Class  16
        (0, 64, 0),  # Class  17
        (64, 128, 128),  # Class 18
        (192, 128, 0),  # Class  19
        (128, 0, 128),  # Class  20
    ]
)


# Function to return all files with a certain ending
def return_files_in_directory(path, ending):
    """Returns all files with a certain ending in path

    Args:
        path (string): Path to files
        ending (string): file ending e.g. .png

    Returns:
        list: list of file paths
    """
    file_paths = []
    for root, dir, files in os.walk(path, topdown=False):
        for file in files:
            file_paths.append(os.path.join(root, file))
    filtered_files = [file for file in file_paths if ending in file]
    return filtered_files


# Function to copy image to new directory
def copy_image(path, new_path):
    shutil.copy(path, new_path)


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
    """Creates a dataset based on a tag and a condition of that tag

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
    # Read into dataframe
    data = {"data": dataset}
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)


def import_dataset(input_path):
    df = pd.read_csv(input_path)
    return df.values.tolist()

# Check pattern of bounding box coordinates
def check_box_coordinates(box_coordinate):
    if "." in box_coordinate:
        return int(float(box_coordinate))
    else:
        return int(box_coordinate)


# Function to return bounding box information of annotation file -> Process dataset
def get_bounding_boxes(xml_file: str):
    """Returns bounding box information of annotation file

    Args:
        xml_file (str): path to file

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


# Function to extract class information of all categories present in the dataset, build csv holding class and color information and export csv
def create_class_color_code_csv(
    dataset, output_path=None, background=False, return_dataframe=False
):
    """Creates a csv that has all unique categories in the dataset and
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
    # set seed to ensure reproducability
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
    category_data="/Users/michaelgroeger/workspace/FEA_Internship/data/VOCdevkit/VOC2012/outputs/color_codes/color_codes_pascal_voc.csv",
    output_path="/Users/michaelgroeger/workspace/FEA_Internship/data/VOCdevkit/VOC2012/outputs/masks",
):
    if isinstance(category_data, str):
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


# All delivered segmentation masks have a white boundary which we want to eliminate
# the color code is [224, 224, 192]
def drop_color_in_image(
    path,
    color_code=[224, 224, 192],
    output_path="/Users/michaelgroeger/workspace/FEA_Internship/data/VOCdevkit/VOC2012/outputs/segmentation_masks",
):
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


# Helper function to convert the masks
def rgb_to_mask(mask_path, color_map, device=None):
    """
    Converts a RGB image mask of shape [batch_size, h, w, 3] to Mask of shape [h, w]. If the image is
    JPG please make sure that there are no additional colors on the mask than the one present in the mapping. You
    can check this by passing the image as a numpy array to np.unique(mask).
    Parameters:
        img: path to RGB image
        color_map: Dictionary representing color mappings
    returns:
        out: A Binary Mask of shape [h, w] as numpy array
    """
    image = Image.open(mask_path)
    # Template for outgoing mask
    if device != None:
        out = torch.zeros([image.size[1], image.size[0]], dtype=np.long, device=torch.device(device))
    else:
        out = torch.zeros([image.size[1], image.size[0]], dtype=np.long)
    # Iterate over pixels of image
    for j in range(image.size[0] - 1):
        for z in range(image.size[1] - 1):
            current_pixel = image.getpixel((j, z))
            if current_pixel in color_map:
                out[z, j] = color_map[current_pixel]
            else:
                out[z, j] = 0
    return out


# This is another helper function to visualize the masks. It was found in the semantic segmentation
# tutorial from captum under https://captum.ai/tutorials/Segmentation_Interpret
# It should only be used for the output of the neural network.


def decode_segmap(image, label_colors, nc=21):
    """Will create a rgb image from the class index mask.

    Args:
        image (numpy array): Image encoded as numpy array.
        label_colors (numpy array): Numpy array holding the labeling colors. Index should be the same like the class index. e.g. label_colors = np.array( [(0, 0, 0),  # Class 0 (192, 128, 128),  # Class  1 ...]
        nc (int, optional): Number of classes. Defaults to 21.

    Returns:
        [numpy array]: Returns numpy array in rgb format. Can be read by PIL.
    """
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


# Helper function to get the classes from a mask
def get_classes_from_mask(mask, class_list=None):
    """Will take as input a mask and a optional list of class names. The class names must coincide with the class indexes to make sense.
    It will flatten the mask into a 1-D array and drop all zeros since they are the background class and most often the most occuring class.
    It will then get the mode and either look up the class string in the class list or simply return the class index.

    Args:
        mask (tensor): [description]
        class_list (list, optional): List holding class names as strings. Defaults to None.

    Returns:
        [string or int]: Either the class string or the class index.
    """
    # flatten mask to be able to apply mode
    mask = torch.flatten(mask)
    
    # Drop zeros since they're the background class
    mask = mask[mask != 0]
    if len(mask) == 0:
        return (None, [None])
    # get most common value
    most_common_class = torch.mode(mask)[0].item()
    all_classes = torch.unique(mask)
    # Drop most common one to avoid dublicates
    all_classes = all_classes[all_classes != most_common_class]
    if class_list != None:
        most_common_class = class_list[most_common_class]
        all_classes = [class_list[i] for i in all_classes]
        return (most_common_class, all_classes)
    else:
        # return everything except the first class since it corresponds to background
        return (most_common_class, all_classes)

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()
    
    
def export_visualize(path, **images):
    """PLot images in one row. And exports it."""
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
            batch_mean_iou = 0
            # Unfold the images and calculate prediction for each of them
            for i in range(batchsize):
                current_gt = gt[i, :, :]
                current_pred = torch.argmax(model(image[i, :, :].unsqueeze(0)), dim=1, keepdim=True)[: , -1, :, :]
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
            pred = torch.argmax(model(image), dim=1, keepdim=True)[: , -1, :, :]
            intersection = torch.logical_and(gt, pred) + eps
            union = torch.logical_or(gt, pred) + eps
            iou_score = torch.sum(intersection) / torch.sum(union)
            return iou_score


# Function to visualize datapoint alongside with ground truth mask and prediction mask
def make_prediction_on_image(dataloader, index, model):
    # Load data
    image, label = iter(dataloader).next()
    # Transfer to gpu
    image, label = image.cuda(), label.cuda()
    # forward + backward + optimize
    outputs = model(image)
    # get argmax for iou calculation, and reshape to same size as label
    out_max = torch.argmax(outputs, dim=1, keepdim=True)[:, -1, :, :]
    rgb_pred = decode_segmap(out_max.detach().cpu().squeeze().numpy()[index, :, :])
    rgb_gt_mask = decode_segmap(label.detach().cpu().squeeze().numpy()[index, :, :])
    show_image = image[index, :, :, :].permute(1, 2, 0).cpu()
    visualize(image=show_image, ground_truth_mask=rgb_gt_mask, predicted_mask=rgb_pred)


# What directories to expect
def pull_images_to_directories(root, directories, datasets, select_subset_of_images):
    """Pull images from datasets into new directory to keep the original data separated from any
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


def return_batch_information(image, argmax_prediction, boxmasks, label, index, class_list, label_colors=label_colors, nc=21):
    if image.shape[0] > 1:
        rgb_pred = decode_segmap(
            argmax_prediction.detach().cpu().squeeze().numpy()[index, :, :], label_colors, nc
        )
        rgb_gt_mask = decode_segmap(
            label.detach().cpu().squeeze().numpy()[index, :, :], label_colors, nc
        )
        boxshink_mask= Image.fromarray(decode_segmap(
            boxmasks.detach().cpu().squeeze().numpy()[index, :, :], label_colors, nc
            ))
        show_image = ToPILImage()(image[index,:,:,:].cpu().detach().squeeze())
        class_in_gt_mask = get_classes_from_mask(label[index, :, :], class_list)
        class_in_prediction = get_classes_from_mask(
            argmax_prediction[index, :, :], class_list
        )
        # Ensure same encoding
        background = show_image.convert("RGBA")
        overlay_prediction = boxshink_mask.convert("RGBA")
        # Create new image from overlap and make overlay 50 % transparent
        prediction_with_image = Image.blend(background, overlay_prediction, 0.5)
        visualize(image=show_image, ground_truth=rgb_gt_mask, prediction=rgb_pred, boxshrink_mask=boxshink_mask, image_with_boxshrink=prediction_with_image)
        print(
            f"\nMost common ground truth class {class_in_gt_mask[0]}, all other classes {class_in_gt_mask[1]}"
        )
        print(
            f"\nMost common prediction class {class_in_prediction[0]}, all other classes {class_in_prediction[1]}"
        )
    else:
        rgb_pred = decode_segmap(
            argmax_prediction.detach().cpu().squeeze().numpy(), label_colors, nc
        )
        rgb_gt_mask = decode_segmap(
            label.detach().cpu().squeeze().numpy(), label_colors, nc
        )
        boxshink_mask= Image.fromarray(decode_segmap(
            boxmasks.detach().cpu().squeeze().numpy(), label_colors, nc
            ))
        show_image = ToPILImage()(image.cpu().detach().squeeze())
        class_in_gt_mask = get_classes_from_mask(label, class_list)
        class_in_prediction = get_classes_from_mask(
            argmax_prediction, class_list
        )
        # Ensure same encoding
        background = show_image.convert("RGBA")
        overlay_prediction = boxshink_mask.convert("RGBA")
        prediction_with_image = Image.blend(background, overlay_prediction, 0.5)
        visualize(image=show_image, ground_truth=rgb_gt_mask, prediction=rgb_pred, boxshrink_mask=boxshink_mask, image_with_boxshrink=prediction_with_image)
        print(
            f"\nMost common ground truth class {class_in_gt_mask[0]}, all other classes {class_in_gt_mask[1]}"
        )
        print(
            f"\nMost common prediction class {class_in_prediction[0]}, all other classes {class_in_prediction[1]}"
        )

def investigate_example(model, dataset, dataset_vis, index, class_list, label_colors=label_colors, nc=21):
    model.eval()
    # get device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    image = dataset[index][0].unsqueeze(0).to(device)
    # Get argmax of prediction
    prediction = model(image)
    argmax_prediction = torch.argmax(prediction, dim=1, keepdim=True)[: , -1, :, :]
    # get label
    label = dataset[index][1]
    # get label to resize untransformed image
    size = label.shape
    image = dataset_vis[index][0].resize((size[0], size[1]), resample=Image.NEAREST)
    # Turn class prediction mask into RGB image
    rgb_pred = Image.fromarray(decode_segmap(
        argmax_prediction.detach().cpu().squeeze().numpy(), label_colors, nc
    ))
    # Turn ground truth into rgb image
    rgb_gt_mask = Image.fromarray(decode_segmap(
        label.detach().cpu().squeeze().numpy(), label_colors, nc
    ))
    # show_image = image[index, :, :, :].permute(1, 2, 0).cpu()
    class_in_gt_mask = get_classes_from_mask(label, class_list)
    class_in_prediction = get_classes_from_mask(argmax_prediction, class_list)
    # Ensure same encoding
    background = image.convert("RGBA")
    overlay_prediction = rgb_pred.convert("RGBA")
    overlay_mask = rgb_gt_mask.convert("RGBA")
    # Create new image from overlap and make overlay 50 % transparent
    prediction_with_image = Image.blend(background, overlay_prediction, 0.5)
    visualize(image=image, ground_truth=rgb_gt_mask, prediction=rgb_pred, prediction_on_image=prediction_with_image)
    print(
        f"\nMost common ground truth class {class_in_gt_mask[0]}, all other classes {class_in_gt_mask[1]}"
    )
    print(
        f"\nMost common prediction class {class_in_prediction[0]}, all other classes {class_in_prediction[1]}"
    )



# Function to unpack and flatten a list of lists
# From: https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
def flatten(L):
    for item in L:
        try:
            yield from flatten(item)
        except TypeError:
            yield item
            
            
            
# Export return batch information to path
def export_return_batch_information(
    image,
    argmax_prediction,
    boxmasks,
    label,
    path,
    epoch,
    alpha,
    superpixel_overlap,
    idx,
    class_list,
    label_colors=label_colors,
    nc=21,
):
    rgb_pred = decode_segmap(
        argmax_prediction.detach().cpu().squeeze().numpy(),
        label_colors,
        nc,
    )
    rgb_gt_mask = decode_segmap(
        label.detach().cpu().squeeze().numpy(), label_colors, nc
    )
    boxshink_mask= Image.fromarray(decode_segmap(
        boxmasks.detach().cpu().squeeze().numpy(), label_colors, nc
        ))
    show_image = ToPILImage()(image.cpu().detach().squeeze())
    class_in_gt_mask = get_classes_from_mask(label, class_list)
    class_in_prediction = get_classes_from_mask(argmax_prediction, class_list)
    # Ensure same encoding
    background = show_image.convert("RGBA")
    overlay_prediction = boxshink_mask.convert("RGBA")
    # Create new image from overlap and make overlay 50 % transparent
    prediction_with_image = Image.blend(background, overlay_prediction, 0.5)
    # Build plot_name
    plot_name = str(idx) + '_' + str(epoch) + '_' + str(alpha) + '_' + str(superpixel_overlap)+ '_' + str(class_in_prediction[0]) + '_' + str(class_in_prediction[1]) + '.png'
    # build path 
    final_path = os.path.join(path, plot_name)
    export_visualize(
        path=final_path,
        image=show_image,
        ground_truth=rgb_gt_mask,
        prediction=rgb_pred,
        boxshrink_mask=boxshink_mask,
        image_with_boxshrink=prediction_with_image
    )