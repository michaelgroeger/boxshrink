import numpy as np
import torch
from PIL import Image
from skimage.segmentation import slic
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DEVICE
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


def get_foreground_background_embeddings(
    argmax_prediction_per_class,
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
    hadamard = all_superpixels_mask.to(device) * argmax_prediction_per_class.to(device)
    overlap = (hadamard / class_indx).type(torch.IntTensor)
    # Instantiate base mask
    torch.zeros(overlap.shape)
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
        base = (
            torch.Tensor(org_img).permute(2, 0, 1).clone().cpu() / 255
        )  # train_input.clone().cpu()
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


def get_first_last(indices, placements):
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
    train_input,
    N_SEGMENTS,
    mean_foreground_embedding,
    mean_background_embedding,
    model,
    threshold_embedding=0,
    threshold_closeness=0,
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
        train_input,
        N_SEGMENTS=N_SEGMENTS,
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


def get_embedding_mask_or_box(
    train_input,
    train_label,
    train_org_image,
    model,
    mean_foreground_embedding,
    mean_background_embedding,
    iou_threshold=0.1,
    mask_occupancy_threshold=0.04,
    n_segments=300,
    threshold_embedding=0,
    threshold_closeness=0,
    iter=1,
    device=DEVICE,
):
    embedding_mask, _, _, _ = create_embedding_mask(
        train_label,
        train_org_image,
        train_input,
        N_SEGMENTS=n_segments,
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
                        train_inputs[idx],
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
            for train_inputs, train_labels, train_org_images in tepoch:
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
                        train_inputs[idx],
                        train_labels[idx],
                        train_org_images[idx],
                        model,
                        mean_foreground_embedding,
                        mean_background_embedding,
                    )
                    masks[idx] = pseudomask
                    mask_dict[idx] = mask_name.replace("tif", "pt").split("/")[-1]
    return masks, mask_dict
