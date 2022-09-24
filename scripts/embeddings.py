import torch
from infer_bounding_boxes import get_bbox_coordinates_one_box
from config import DEVICE
from skimage.segmentation import slic
from tqdm import tqdm
from config import MASK_OCCUPANCY_THRESHOLD, DEVICE, IOU_THRESHOLD, N_SEGMENTS
from crf import crf, pass_pseudomask_or_ground_truth
from tools import visualize
from tools import flatten
from tifffile import imread
import numpy as np
from PIL import Image
from dataset import img_transform
import matplotlib.pyplot as plt


class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(ResnetFeatureExtractor, self).__init__()
        self.model = model
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])

    def __call__(self, x):
        return self.feature_extractor(x)[:, :, 0, 0]


def get_cosine_sim_score(feat_1, feat_2, cosine_fct=torch.nn.CosineSimilarity(dim=0)):
    return torch.sum(cosine_fct(feat_1.squeeze(), feat_2.squeeze()))


def get_foreground_background_embeddings(
    initial_mask,
    org_img,
    train_input,
    threshold,
    N_SEGMENTS,
    model,
    class_indx=1,
    compactness=10,
    sigma=1,
    start_label=1,
    device=DEVICE,
):
    # get superpixels
    org_img = org_img.cpu().detach().numpy()
    all_superpixels_mask = torch.from_numpy(
        slic(
            org_img,
            n_segments=N_SEGMENTS,
            compactness=compactness,
            sigma=sigma,
            start_label=start_label,
        )
    )
    hadamard = all_superpixels_mask.to(device) * initial_mask.to(device)
    overlap = (hadamard / class_indx).type(torch.IntTensor)
    # Get numbers to list, start from second element because first is 0
    relevant_superpixels = torch.unique(overlap).int().tolist()[1:]
    relevant_superpixels_thresholded = []
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
            relevant_superpixels_thresholded.append(superpixel)
    background_superpixels = [
        i.item()
        for i in torch.unique(all_superpixels_mask)
        if i not in relevant_superpixels_thresholded
    ]
    foreground_embeddings = torch.zeros([len(relevant_superpixels_thresholded), 2048])
    background_embeddings = torch.zeros([len(background_superpixels), 2048])
    for i, superpixel in enumerate(relevant_superpixels_thresholded):
        all_superpixels_mask_tmp = all_superpixels_mask.clone()
        all_superpixels_mask_tmp[all_superpixels_mask_tmp != superpixel] = 0
        all_superpixels_mask_tmp[all_superpixels_mask_tmp > 0] = 1
        s, l = get_bbox_coordinates_one_box(all_superpixels_mask_tmp)
        base = torch.Tensor(org_img).permute(2, 0, 1).clone().cpu() / 255
        base_aspm = base.clone()
        base_aspm[0, :, :] = base_aspm[0, :, :] * all_superpixels_mask_tmp
        base_aspm[1, :, :] = base_aspm[1, :, :] * all_superpixels_mask_tmp
        base_aspm[2, :, :] = base_aspm[2, :, :] * all_superpixels_mask_tmp
        cut = base_aspm[:, s[1] : l[1], s[0] : l[0]].unsqueeze(0).to(device)
        with torch.no_grad():
            feat_foreground_sp = model(cut)
            foreground_embeddings[i, :] = feat_foreground_sp
    for i, superpixel in enumerate(background_superpixels):
        all_superpixels_mask_tmp = all_superpixels_mask.clone()
        all_superpixels_mask_tmp[all_superpixels_mask_tmp != superpixel] = 0
        all_superpixels_mask_tmp[all_superpixels_mask_tmp > 0] = 1
        s, l = get_bbox_coordinates_one_box(all_superpixels_mask_tmp)
        base = train_input.clone().cpu()
        base_aspm = base.clone()

        base_aspm[0, :, :] = base_aspm[0, :, :] * all_superpixels_mask_tmp
        base_aspm[1, :, :] = base_aspm[1, :, :] * all_superpixels_mask_tmp
        base_aspm[2, :, :] = base_aspm[2, :, :] * all_superpixels_mask_tmp
        cut = base_aspm[:, s[1] : l[1], s[0] : l[0]].unsqueeze(0).to(device)
        with torch.no_grad():
            feat_background_sp = model(cut)
            background_embeddings[i, :] = feat_background_sp
    return (
        foreground_embeddings,
        background_embeddings,
        relevant_superpixels_thresholded,
        all_superpixels_mask,
    )


def get_mean_embeddings(
    data_loader,
    model,
    embedding_dir,
    get_foreground_background_embeddings=get_foreground_background_embeddings,
    N_SEGMENTS=250,
    THRESHOLD=0.1,
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
                    train_inputs.to(DEVICE),
                    train_labels.to(DEVICE),
                    train_org_images.to(DEVICE),
                )
                for i in range(0, train_inputs.shape[0], 1):
                    embed_f, embed_b, _, _ = get_foreground_background_embeddings(
                        train_labels[i],
                        train_org_images[i],
                        train_inputs[i],
                        N_SEGMENTS=N_SEGMENTS,
                        threshold=THRESHOLD,
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
    threshold,
):
    close_f_foreground_embeddings = []
    for i in range(foreground_embeddings.shape[0]):
        f_cos = cosine_fct(mean_foreground_embedding, foreground_embeddings[i])
        b_cos = cosine_fct(mean_background_embedding, foreground_embeddings[i])
        diff = abs(b_cos - f_cos)
        if (f_cos > b_cos or f_cos == b_cos) and diff <= threshold:
            close_f_foreground_embeddings.append(relevant_superpixels_thresholded[i])
    return close_f_foreground_embeddings


def scan_outer_boundary(superpixel_mask, flatten_list_fct=flatten):
    # We will collect those superpixels which are on the outer boundary so they
    # have a zero neighbour.
    superpixel_mask_refined = superpixel_mask.clone()
    outer_superpixel_rows, outer_superpixel_cols, outer_all = [], [], []
    tuples = torch.nonzero(superpixel_mask_refined)
    rows = torch.unique(tuples[:, 0])
    columns = torch.unique(tuples[:, 1])
    # scan over rows
    for i in rows:
        current_row = superpixel_mask_refined[i, :]
        unique_non_zeroed_row = torch.unique(
            current_row[current_row.nonzero(as_tuple=True)], sorted=False
        )
        first_superpixel = unique_non_zeroed_row[-1]
        last_superpixel = unique_non_zeroed_row[0]
        outer_superpixel_rows.append(first_superpixel.item())
        outer_superpixel_rows.append(last_superpixel.item())
    # scan over columns
    for i in columns:
        current_column = superpixel_mask_refined[:, i]
        unique_non_zeroed_column = torch.unique(
            current_column[current_column.nonzero(as_tuple=True)], sorted=False
        )
        first_superpixel = unique_non_zeroed_column[-1]
        last_superpixel = unique_non_zeroed_column[0]
        outer_superpixel_cols.append(first_superpixel.item())
        outer_superpixel_cols.append(last_superpixel.item())
    outer_all.append(outer_superpixel_cols)
    outer_all.append(outer_superpixel_rows)
    outer_all = flatten(outer_all)
    return list(set(outer_all))


def create_embedding_mask(
    train_label,
    train_org_image,
    train_input,
    N_SEGMENTS,
    mean_foreground_embedding,
    mean_background_embedding,
    threshold_embedding=0,
    threshold_closeness=0,
    cosine_function=get_cosine_sim_score,
    assign_label_based_on_closeness=assign_foreground_sp,
    get_foreground_background_embeddings_function=get_foreground_background_embeddings,
    scan_outer_pixels=True,
    scan_outer_superpixels_function=scan_outer_boundary,
    postprocess_crf=True,
    iter=1,
):
    (
        foreground_embeddings,
        background_embeddings,
        relevant_superpixels,
        all_superpixels_mask,
    ) = get_foreground_background_embeddings_function(
        train_label,
        train_org_image,
        train_input,
        N_SEGMENTS=N_SEGMENTS,
        threshold=threshold_embedding,
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
                threshold=threshold_closeness,
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
