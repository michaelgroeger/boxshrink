import numpy as np
import torch
from PIL import Image
from tools import return_files_in_directory
from torchvision.transforms import ToTensor
from tqdm import tqdm


def span_columns(max_x, max_y):
    """Span matrix that will be used to infer the column coordinates of a box.
    Column 1 will be all ones, column 2 all 2s and so on

    Args:
        max_x (int): maximum coordinate along axis x
        max_y (int): maximum coordinate along axis y

    Returns:
        tensor
    """
    cols = torch.zeros([max_y, max_x])
    for i in range(max_x):
        cols[:, i] = i + 1
    return cols


def span_rows(max_x, max_y):
    """Span matrix that will be used to infer the row coordinates of a box.
    row 1 will be all ones, row 2 all 2s and so on

    Args:
        max_x (int): maximum coordinate along axis x
        max_y (int): maximum coordinate along axis y

    Returns:
        tensor
    """
    rows = torch.zeros([max_y, max_x])
    for i in range(max_y):
        rows[i, :] = i + 1
    return rows


def get_bbox_coordinates_one_box(tensor):
    """Return coordinates in case there is just one box present on the mask.

    Args:
        tensor (_type_): Mask tensor

    Returns:
        Sets: Two sets holding coordinate information
    """
    all_x, all_y = (tensor.squeeze() == 1).nonzero(as_tuple=True)
    smallest_x, smallest_y = torch.min(all_x).item(), torch.min(all_y).item()
    largest_x, largest_y = torch.max(all_x).item(), torch.max(all_y).item()
    return (smallest_y, smallest_x), (largest_y, largest_x)


def get_min_max_x(unique_tensor):
    unique_tensor = unique_tensor[unique_tensor != 0]
    min_xs = []
    min_xs.append(unique_tensor[0].item())
    max_xs = []
    spotted_more_than_one_box = False
    for i in range(len(unique_tensor)):
        next_idx = i + 1
        if next_idx <= (len(unique_tensor) - 1):
            next_element = unique_tensor[next_idx]
            current_element = unique_tensor[i]
            diff = next_element - current_element
            if diff > 1:
                spotted_more_than_one_box = True
                min_xs.append(next_element.item())
                max_xs.append(current_element.item())
        else:
            current_element = unique_tensor[i]
            max_xs.append(current_element.item())
    return min_xs, max_xs, spotted_more_than_one_box


def get_min_max_y(base_tensor, min_xs, max_xs):
    ys = base_tensor[1]
    min_ys, max_ys = [], []
    # Slice tensor and extract values
    for idx, i in enumerate(min_xs):
        # slice colum wise
        start = int(i - 1)
        end = int(max_xs[idx])
        slice = ys[:, start:end]
        # get unique values
        unique_tensor = torch.unique(slice)
        unique_tensor = unique_tensor[unique_tensor != 0]
        min_ys.append(unique_tensor[0].item())
        for i in range(len(unique_tensor)):
            next_idx = i + 1
            if next_idx <= (len(unique_tensor) - 1):
                next_element = unique_tensor[next_idx]
                current_element = unique_tensor[i]
                diff = next_element - current_element
                if diff > 1:
                    min_ys.append(next_element.item())
                    max_ys.append(current_element.item())
            else:
                current_element = unique_tensor[i]
                max_ys.append(current_element.item())
    return min_ys, max_ys


def get_bbox_coordinates(tensor):
    """
    Will check if there are more than one boxes on the mask and return the
    coordinates.

    Args:
        tensor (tensor): Segmentation mask Mask

    Returns:
        sets, bool=optional: (min_y, min_x) (max_y, max_x), bool indicating if there is more than one box
    """
    # get base
    tensor = tensor.squeeze()
    n_rows = tensor.shape[0]
    n_cols = tensor.shape[1]
    cols = span_columns(n_cols, n_rows)
    rows = span_rows(n_cols, n_rows)
    base = torch.zeros([2, n_rows, n_cols])
    base[0, :, :] = cols
    base[1, :, :] = rows
    res = tensor * base
    min_xs, max_xs, spotted_more_than_one_box = get_min_max_x(torch.unique(res[0]))
    if spotted_more_than_one_box == True:
        min_ys, max_ys = get_min_max_y(res, min_xs, max_xs)
        # build min
        if len(min_xs) < len(min_ys):
            min_xs = [min_xs[-1] for i in range(len(min_ys))]
            max_xs = [max_xs[-1] for i in range(len(min_ys))]
        if len(min_ys) < len(min_xs):
            min_ys = [min_ys[-1] for i in range(len(min_xs))]
            max_ys = [max_ys[-1] for i in range(len(min_xs))]
        coord = [
            ((min_xs[i], min_ys[i]), (max_xs[i], max_ys[i])) for i in range(len(min_xs))
        ]
        return coord, spotted_more_than_one_box
    else:
        return get_bbox_coordinates_one_box(tensor)


def generate_mask_from_box_colonoscopy(
    image,
    smallest,
    largest,
):
    """Function to generate the box-like segmentation masks to be used during training

    Args:
        image (tensor): csv holding path to each considered xml file
        smallest (set): min_y, min_x coordinate of box
        largest (set): max_y, max_x coordinate of box
    """
    image_width, image_height = image.shape[2], image.shape[1]
    # Create mask
    mask = np.zeros([image_height, image_width], dtype=np.uint8)
    # Get all boxes present on the image
    # get coordinates
    ymin, xmin, ymax, xmax = (smallest[0], smallest[1], largest[0], largest[1])
    # Python does array[inclusive:exclusive] therefore we need to add + 1 to the max values
    # But then we also need to catch the case that the +1 will be out of the image
    if ymax != image_height:
        ymax = ymax + 1
    if xmax != image_width:
        xmax = xmax + 1
    mask[xmin:xmax, ymin:ymax] = 1
    mask = Image.fromarray(np.uint8(mask * 255), "L")
    return mask


def infer_all_bboxes(mask_dir, box_dir):
    """Infers bounding box masks for all files in directory and returns them to target directory

    Args:
        mask_dir (string)
        box_dir (string)
    """
    # get all masks from dir
    files = return_files_in_directory(mask_dir, ending=".tif")
    for _, mask_path in tqdm(enumerate(files), desc="Creating Boxes from masks"):
        # load image
        image = ToTensor()(Image.open(mask_path))
        # map all positive values to 1
        image[image > 0] = 1.0
        # get box coordinates
        smallest, largest = get_bbox_coordinates(image)
        # infer mask
        bb_mask = generate_mask_from_box_colonoscopy(
            image=image, smallest=smallest, largest=largest
        )
        # save mask to new dir
        output_path_mask = (box_dir + "/" + mask_path.split("/")[-1]).replace(
            "tif", "png"
        )
        bb_mask.save(output_path_mask, quality=100, subsampling=0)
