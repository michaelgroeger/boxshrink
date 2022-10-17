#############################################################################################################################
#                                                   Helper functions                                                        #
#############################################################################################################################
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms import ToPILImage


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


# From: https://captum.ai/tutorials/Segmentation_Interpret
def decode_segmap(image, label_colors, nc):
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
    """Will take as input a mask and an optional list of class names. The class names must coincide with the class indexes to make sense.
    It will flatten the mask into a 1-D array and drop all zeros since they are the background class and most often the most occuring class.
    It will then get the mode and either look up the class string in the class list or simply return the class index.

    Args:
        mask (tensor): tensor of the segmentation mask
        class_list (list, optional): List holding class names as strings. Defaults to None.

    Returns:
        [set]: Set of either the class strings or the class indexes.
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


# Helper function to convert the masks
def rgb_to_mask(mask_path, color_map, device=None):
    """
    Converts segmentation mask image to class index map to be fed into the network
    Parameters:
        img: path to RGB image
        color_map: Dictionary representing color mappings
    returns:
        out: A Binary Mask of shape [h, w] as numpy array
    """
    image = Image.open(mask_path)
    # Template for outgoing mask
    if device != None:
        out = torch.zeros(
            [image.size[1], image.size[0]], dtype=np.long, device=torch.device(device)
        )
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


def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()


def return_batch_information(
    org_img, argmax_prediction, label, index, class_list, label_colors
):
    """Returns image, ground truth and current prediction of one element at index at the batch for debugging purposes.

    Args:
        org_img (tensor): original, unnormalized image
        argmax_prediction (tensor): argmaxed model output for batch
        label (tensor): ground truth label
        index (_type_): index of element in batch
        class_list (_type_, optional): _description_. Defaults to CLASSES.
        label_colors (_type_, optional): _description_. Defaults to label_colors.
    """
    nc = len(class_list)
    if org_img.shape[0] > 1:
        rgb_pred = Image.fromarray(
            decode_segmap(argmax_prediction[index, :, :].squeeze(), label_colors, nc)
        )
        label_image = Image.fromarray(
            decode_segmap(
                label[index, :, :].detach().cpu().squeeze().numpy(), label_colors, nc
            )
        )
        show_image = ToPILImage()(
            org_img[index, :, :, :].permute(2, 0, 1).cpu().detach().squeeze()
        )
        # Ensure same encoding
        background = show_image.convert("RGBA")
        overlaylabel = label_image.convert("RGBA")
        overlaypred = rgb_pred.convert("RGBA")
        label_with_image = Image.blend(background, overlaylabel, 0.5)
        # Create new image from overlap and make overlay 50 % transparent
        prediction_with_image = Image.blend(background, overlaypred, 0.5)
        visualize(
            org_img=show_image,
            prediction_with_image=prediction_with_image,
            label_with_image=label_with_image,
        )


# For sorting images and masks
# https://nedbatchelder.com/blog/200712/human_sorting.html
def tryint(s):
    """
    Return an int if possible, or `s` unchanged.
    """
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    """
    Turn a string into a list of string and number chunks.

    >>> alphanum_key("z23a")
    ["z", 23, "a"]

    """
    return [tryint(c) for c in re.split("([0-9]+)", s)]


def human_sort(l):
    """
    Sort a list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def overlay_mask_on_image(org_image, cam):
    rgb_img = np.float32(org_image) / 255
    cam_rgb = show_cam_on_image(rgb_img, cam, use_rgb=True)
    return cam_rgb


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


label_colors = np.array([(0, 0, 0), (128, 128, 128)])
