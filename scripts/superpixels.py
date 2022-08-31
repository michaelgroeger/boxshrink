from numpy.core.numeric import count_nonzero
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import torch


def create_superpixel_mask(argmax_prediction_per_class, image, threshold = 0.50, class_indx=1,  N_SEGMENTS=200, compactness=10, sigma=1, start_label=1, device=device):
    # get superpixels
    image = image.cpu().detach().numpy()
    all_superpixels_mask = torch.from_numpy(slic(image, n_segments=N_SEGMENTS, compactness=compactness, sigma=sigma, start_label=start_label))
    hadamard = all_superpixels_mask.to(device) * argmax_prediction_per_class.to(device)
    overlap = (hadamard / class_indx).type(torch.IntTensor)
    # Instantiate base mask
    base_mask = torch.zeros(overlap.shape)
    # Get numbers to list, start from second element because first is 0 
    relevant_superpixels = torch.unique(overlap).int().tolist()[1:]
    for superpixel in relevant_superpixels:
        temp = overlap.clone()
        org = all_superpixels_mask.clone()
        # Check how many are non-zero in superpixel mask
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

def precomputed_create_superpixel_mask(argmax_prediction_per_class, boundaries, threshold = 0.50, class_indx=4):
    # get superpixels
    with torch.no_grad():
        pred_cleaned = argmax_prediction_per_class.squeeze(0).clone()
        pred_cleaned[pred_cleaned != class_indx] = 0
        hadamard = boundaries * pred_cleaned
        overlap = (hadamard / class_indx).type(torch.IntTensor)
        # Instantiate base mask
        base_mask = torch.zeros(overlap.shape).to(DEVICE)
        # Get numbers to list, start from second element because first is 0 
        relevant_superpixels = torch.unique(overlap).int().tolist()[1:]
        for superpixel in relevant_superpixels:
            temp = overlap.clone()
            org = boundaries.clone()
            # Check how many are non-zero in superpixel mask
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
        base_mask[(base_mask != 0) & (base_mask != class_indx)] = class_indx
        return base_mask.type(torch.IntTensor)

def visualize_superpixels(boundaries, **images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(25, 10))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(mark_boundaries(image, boundaries))
    plt.show()

def export_superpixel_crf_masks_for_train_data(dataset, export_path="path"):
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
            sp_mask = create_superpixel_mask(mask, img, N_SEGMENTS=200, threshold=0.60)
            img, sp_mask = img.to(device), sp_mask.to(device)
            pseudomask = process_batch_crf(img, sp_mask)
            pseudomask = pass_pseudomask_or_ground_truth(mask.to(device), pseudomask)
            pseudomask = Image.fromarray(np.uint8(pseudomask.cpu().detach() * 255) , 'L')
            output_path_mask = (
            export_path + "/" + _.split('/')[-1]
            ).replace("tif", "png")
            pseudomask.save(output_path_mask, quality=100, subsampling=0)
