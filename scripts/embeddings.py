import numpy as np
import torch
from PIL import Image
from skimage.segmentation import slic
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.config import (
    DEVICE,
    IOU_THRESHOLD,
    MASK_OCCUPANCY_THRESHOLD,
    N_SEGMENTS_ROBUST,
    SLIC_COMPACTNESS,
    SUPERPIXEL_OVERLAP_THRESHOLD,
    THRESHOLD_CLOSNESS,
)
from scripts.crf import crf, pass_pseudomask_or_ground_truth
from scripts.infer_bounding_boxes import get_bbox_coordinates_one_box


class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(ResnetFeatureExtractor, self).__init__()
        self.model = model
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])

    def __call__(self, x):
        return self.feature_extractor(x)[:, :, 0, 0]


def get_cosine_sim_score(feat_1, feat_2, cosine_fct=torch.nn.CosineSimilarity(dim=0)):
    return torch.sum(cosine_fct(feat_1.squeeze(), feat_2.squeeze()))


def get_bbox_coordinates_one_box(tensor):
    all_x, all_y = (tensor.squeeze() == 1).nonzero(as_tuple=True)
    smallest_x, smallest_y = torch.min(all_x).item(), torch.min(all_y).item()
    largest_x, largest_y = torch.max(all_x).item(), torch.max(all_y).item()
    return (smallest_y, smallest_x), (largest_y, largest_x)


def fill_embedding_matrix(
    embedding_matrix,
    base_mask,
    all_superpixels_mask,
    relevant_superpixels,
    model,
    device=DEVICE,
):
    for i, superpixel in enumerate(relevant_superpixels):
        all_superpixels_mask_tmp = all_superpixels_mask.clone()
        all_superpixels_mask_tmp[all_superpixels_mask_tmp != superpixel] = 0
        all_superpixels_mask_tmp[all_superpixels_mask_tmp > 0] = 1
        smallest, largest = get_bbox_coordinates_one_box(all_superpixels_mask_tmp)
        base_all_superpixels_mask = base_mask.clone()
        base_all_superpixels_mask[0, :, :] = (
            base_all_superpixels_mask[0, :, :] * all_superpixels_mask_tmp
        )
        base_all_superpixels_mask[1, :, :] = (
            base_all_superpixels_mask[1, :, :] * all_superpixels_mask_tmp
        )
        base_all_superpixels_mask[2, :, :] = (
            base_all_superpixels_mask[2, :, :] * all_superpixels_mask_tmp
        )
        cut = (
            base_all_superpixels_mask[
                :, smallest[1] : largest[1], smallest[0] : largest[0]
            ]
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            superpixel_embedding = model(cut)
            embedding_matrix[i, :] = superpixel_embedding
    return embedding_matrix


def get_foreground_background_embeddings(
    mask,
    org_img,
    model,
    threshold=SUPERPIXEL_OVERLAP_THRESHOLD,
    n_segments=N_SEGMENTS_ROBUST,
    class_indx=1,
    compactness=SLIC_COMPACTNESS,
    sigma=1,
    start_label=1,
    device=DEVICE,
):
    # get superpixels
    all_superpixels_mask = torch.from_numpy(
        slic(
            org_img.cpu().detach().numpy(),
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            start_label=start_label,
        )
    )
    hadamard = all_superpixels_mask.to(device) * mask.to(device)
    overlap = (hadamard / class_indx).type(torch.IntTensor)
    # Get numbers to list, start from second element because first is 0
    relevant_superpixels = torch.unique(overlap).int().tolist()[1:]
    foreground_superpixels = []
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
            foreground_superpixels.append(superpixel)
    background_superpixels = [
        i.item()
        for i in torch.unique(all_superpixels_mask)
        if i not in foreground_superpixels
    ]
    foreground_embeddings = torch.zeros([len(foreground_superpixels), 2048])
    background_embeddings = torch.zeros([len(background_superpixels), 2048])
    base = torch.Tensor(org_img).permute(2, 0, 1).clone().cpu() / 255
    foreground_embeddings = fill_embedding_matrix(
        foreground_embeddings,
        base,
        all_superpixels_mask,
        foreground_superpixels,
        model,
        device=DEVICE,
    )
    background_embeddings = fill_embedding_matrix(
        background_embeddings,
        base,
        all_superpixels_mask,
        background_superpixels,
        model,
        device=DEVICE,
    )
    return (
        foreground_embeddings,
        background_embeddings,
        foreground_superpixels,
        all_superpixels_mask,
    )


def get_mean_embeddings(
    data_loader,
    model,
    embedding_dir,
    get_foreground_background_embeddings=get_foreground_background_embeddings,
    n_segments=N_SEGMENTS_ROBUST,
    threshold=SUPERPIXEL_OVERLAP_THRESHOLD,
    device=DEVICE,
):
    foreground_embeddings = torch.zeros([len(data_loader.dataset), 2048])
    background_embeddings = torch.zeros([len(data_loader.dataset), 2048])
    counter = 0
    for epoch in range(0, 1):
        batch = 0
        with tqdm(data_loader, unit="batch") as tepoch:
            for train_inputs, train_labels, train_org_images in tepoch:
                batch += 1
                tepoch.set_description(f"Epoch {epoch}")
                train_inputs, train_labels, train_org_images = (
                    train_inputs.to(device),
                    train_labels.to(device),
                    train_org_images.to(device),
                )
                for i in range(0, train_inputs.shape[0], 1):
                    embed_f, embed_b, _, _ = get_foreground_background_embeddings(
                        train_labels[i],
                        train_org_images[i],
                        n_segments=n_segments,
                        threshold=threshold,
                        model=model,
                    )
                    mean_f = torch.mean(embed_f, dim=0)
                    mean_b = torch.mean(embed_b, dim=0)
                    foreground_embeddings[counter, :] += mean_f.cpu()
                    background_embeddings[counter, :] += mean_b.cpu()
                    counter += 1
    overall_foreground_mean = torch.mean(foreground_embeddings, dim=0)
    overall_background_mean = torch.mean(background_embeddings, dim=0)
    torch.save(overall_foreground_mean, embedding_dir + "foreground.pt")
    torch.save(overall_background_mean, embedding_dir + "background.pt")


def assign_foreground_sp(
    cosine_fct,
    mean_foreground_embedding,
    mean_background_embedding,
    relevant_superpixels_thresholded,
    foreground_embeddings,
    threshold_closeness,
):
    close_f_foreground_embeddings = []
    for i in range(foreground_embeddings.shape[0]):
        f_cos = cosine_fct(mean_foreground_embedding, foreground_embeddings[i])
        b_cos = cosine_fct(mean_background_embedding, foreground_embeddings[i])
        diff = abs(b_cos - f_cos)
        if (f_cos > b_cos or f_cos == b_cos) and diff <= threshold_closeness:
            close_f_foreground_embeddings.append(relevant_superpixels_thresholded[i])
    return close_f_foreground_embeddings


def get_first_last(indices, placements):
    """Returns the first and last superpixel value of one row in the superpixel mask.

    Args:
        indices (tensor): indexes in which the values are placed
        placements (tensor): rows | cols

    Returns:
        dict: holding row | column and the index of the first and last non-zero entry
    """
    unique_indices = torch.unique(indices)
    first_last_per_index = {}
    for i in unique_indices:
        # get corresponding index in indices
        placement = placements[(indices == i).nonzero(as_tuple=True)[0]]
        first, last = placement[0], placement[-1]
        first_last_per_index[i.item()] = list(set([first.item(), last.item()]))
    return first_last_per_index


def lookup_values(first_last_per_index, tensor, mode):
    keys = list(first_last_per_index.keys())
    values = []
    assert mode in ["rows", "cols"], "Please provide mode one of 'rows', 'cols'"
    if mode == "rows":
        for key in keys:
            for i in tensor[key, first_last_per_index[key]].tolist():
                values.append(int(i))
    elif mode == "cols":
        for key in keys:
            for i in tensor[first_last_per_index[key], key].tolist():
                values.append(int(i))
    return values


def get_relevant_values(values_rows, values_cols):
    return list(set(values_rows + values_cols))


def get_outer_superpixels(superpixel_mask):
    rows, columns = superpixel_mask.nonzero(as_tuple=True)
    res_1 = lookup_values(get_first_last(rows, columns), superpixel_mask, mode="rows")
    res_2 = lookup_values(get_first_last(columns, rows), superpixel_mask, mode="cols")
    return get_relevant_values(res_1, res_2)


def create_embedding_mask(
    train_label,
    train_org_image,
    mean_foreground_embedding,
    mean_background_embedding,
    model,
    n_segments=N_SEGMENTS_ROBUST,
    threshold_embedding=SUPERPIXEL_OVERLAP_THRESHOLD,
    threshold_closeness=THRESHOLD_CLOSNESS,
    cosine_function=get_cosine_sim_score,
    get_foreground_background_embeddings_function=get_foreground_background_embeddings,
    scan_outer_pixels=True,
    scan_outer_superpixels_function=get_outer_superpixels,
    postprocess_crf=True,
    iter=1,
):
    (
        foreground_embeddings,
        _,
        relevant_superpixels,
        all_superpixels_mask,
    ) = get_foreground_background_embeddings_function(
        train_label,
        train_org_image,
        n_segments=n_segments,
        threshold=threshold_embedding,
        model=model,
    )
    for i in range(0, iter, 1):
        not_in_relevant_superpixels = [
            i
            for i in torch.unique(all_superpixels_mask)
            if i not in relevant_superpixels
        ]
        embedding_mask_relevant_superpixels = all_superpixels_mask.clone()
        for i in not_in_relevant_superpixels:
            embedding_mask_relevant_superpixels[
                embedding_mask_relevant_superpixels == i
            ] = 0
        tmp = embedding_mask_relevant_superpixels.clone()
        tmp[tmp > 0] = 1
        if scan_outer_pixels == True:
            outer_superpixels = scan_outer_superpixels_function(
                embedding_mask_relevant_superpixels
            )
            indexes_outer = [relevant_superpixels.index(i) for i in outer_superpixels]
            outer_foreground_embedding = foreground_embeddings[indexes_outer, :]
            close_foreground_outer_superpixels = assign_foreground_sp(
                cosine_function,
                mean_foreground_embedding,
                mean_background_embedding,
                relevant_superpixels_thresholded=outer_superpixels,
                foreground_embeddings=outer_foreground_embedding,
                threshold_closeness=threshold_closeness,
            )
            to_be_dropped = [
                i
                for i in outer_superpixels
                if i not in close_foreground_outer_superpixels
            ]
            relevant_superpixels = [
                i for i in relevant_superpixels if i not in to_be_dropped
            ]
            not_in_relevant_superpixels = [
                i
                for i in torch.unique(all_superpixels_mask)
                if i not in relevant_superpixels
            ]
            for i in not_in_relevant_superpixels:
                embedding_mask_relevant_superpixels[
                    embedding_mask_relevant_superpixels == i
                ] = 0
    embedding_mask_relevant_superpixels[embedding_mask_relevant_superpixels > 0] = 1
    if postprocess_crf == True:
        embedding_mask_relevant_superpixels = crf(
            train_org_image, embedding_mask_relevant_superpixels
        )
    else:
        embedding_mask_relevant_superpixels[embedding_mask_relevant_superpixels > 0] = 1
        if postprocess_crf == True:
            embedding_mask_relevant_superpixels = crf(
                train_org_image, embedding_mask_relevant_superpixels
            )

    return (
        embedding_mask_relevant_superpixels,
        outer_superpixels,
        embedding_mask_relevant_superpixels,
        all_superpixels_mask,
    )


def get_embedding_mask_or_box(
    train_label,
    train_org_image,
    model,
    mean_foreground_embedding,
    mean_background_embedding,
    iou_threshold=IOU_THRESHOLD,
    mask_occupancy_threshold=MASK_OCCUPANCY_THRESHOLD,
    n_segments=N_SEGMENTS_ROBUST,
    threshold_embedding=SUPERPIXEL_OVERLAP_THRESHOLD,
    threshold_closeness=THRESHOLD_CLOSNESS,
    iter=1,
    device=DEVICE,
):
    embedding_mask, _, _, _ = create_embedding_mask(
        train_label,
        train_org_image,
        n_segments=n_segments,
        threshold_embedding=threshold_embedding,
        iter=iter,
        threshold_closeness=threshold_closeness,
        model=model,
        mean_foreground_embedding=mean_foreground_embedding,
        mean_background_embedding=mean_background_embedding,
    )
    return pass_pseudomask_or_ground_truth(
        train_label.to(device),
        embedding_mask.to(device),
        iou_threshold=iou_threshold,
        mask_occupancy_threshold=mask_occupancy_threshold,
    )


def generate_embedding_masks_for_dataset(
    dataset,
    export_path,
    model,
    mean_foreground_embedding,
    mean_background_embedding,
    save_as_png=True,
):
    # No shuffle because we simply want to generate masks to be used in later training
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    filecounter = 0
    for epoch in range(0, 1):
        with tqdm(
            data_loader, unit="batch", desc="Generating Embedding Masks"
        ) as tepoch:
            for train_inputs, train_labels, train_org_images in tepoch:
                for idx in range(train_labels.shape[0]):
                    mask_name = dataset.X[filecounter]
                    pseudomask = get_embedding_mask_or_box(
                        train_labels[idx],
                        train_org_images[idx],
                        model,
                        mean_foreground_embedding,
                        mean_background_embedding,
                    )
                    if save_as_png == True:
                        output_path_mask = (
                            export_path + "/" + mask_name.split("/")[-1]
                        ).replace("tif", "png")
                        pseudomask = Image.fromarray(
                            np.uint8(pseudomask.cpu().detach() * 255), "L"
                        )
                        pseudomask.save(output_path_mask, quality=100, subsampling=0)
                    else:
                        output_path_mask = (
                            export_path + "/" + mask_name.split("/")[-1]
                        ).replace("tif", "pt")
                        torch.save(pseudomask, output_path_mask)
                    filecounter += 1


def return_embedding_masks_for_dataset(
    dataset, model, mean_foreground_embedding, mean_background_embedding
):
    # No shuffle because we simply want to generate masks to be used in later training
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    mask_dict = {}
    for epoch in range(0, 1):
        with tqdm(
            data_loader, unit="batch", desc="Generating Embedding Masks"
        ) as tepoch:
            for _, train_labels, train_org_images in tepoch:
                masks = torch.zeros(
                    [
                        train_labels.shape[0],
                        train_labels.shape[1],
                        train_labels.shape[2],
                    ],
                    dtype=train_labels.dtype,
                    layout=train_labels.layout,
                    device=train_labels.device,
                )
                for idx in range(train_labels.shape[0]):
                    mask_name = dataset.X[idx]
                    pseudomask = get_embedding_mask_or_box(
                        train_labels[idx],
                        train_org_images[idx],
                        model,
                        mean_foreground_embedding,
                        mean_background_embedding,
                    )
                    masks[idx] = pseudomask
                    mask_dict[idx] = mask_name.replace("tif", "pt").split("/")[-1]
    return masks, mask_dict
