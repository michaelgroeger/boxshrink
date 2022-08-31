#############################################################################################################################
#                                           Helper functions used to create the paper                                       #
#############################################################################################################################
import numpy as np
import os
import torch 
from PIL import Image
import matplotlib.pyplot as plt
from voc_tools import (
    return_files_in_directory,
    decode_segmap,
    get_classes_from_mask,
    rgb_to_mask,
    visualize,
    return_batch_information,
    flatten
    )
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


# Function to unpack and flatten a list of lists
# From: https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
def flatten(L):
    for item in L:
        try:
            yield from flatten(item)
        except TypeError:
            yield item
