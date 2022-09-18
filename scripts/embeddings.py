import torch
from infer_bounding_boxes import get_bbox_coordinates_one_box
from superpixels import visualize_superpixels
from tools import visualize
from config import DEVICE
from skimage.segmentation import slic
from tqdm import tqdm

class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(ResnetFeatureExtractor, self).__init__()
        self.model = model
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
                
    def __call__(self, x):
        return self.feature_extractor(x)[:, :, 0, 0]

def get_cosine_sim_score(feat_1, feat_2, cosine_fct=torch.nn.CosineSimilarity(dim=0)):
    return (torch.sum(cosine_fct(feat_1.squeeze(), feat_2.squeeze())))

def get_foreground_background_embeddings(initial_mask, org_img, train_input, threshold, N_SEGMENTS, model, class_indx=1,  compactness=10, sigma=1, start_label=1, device=DEVICE):
    # get superpixels
    org_img = org_img.cpu().detach().numpy()
    all_superpixels_mask = torch.from_numpy(slic(org_img, n_segments=N_SEGMENTS, compactness=compactness, sigma=sigma, start_label=start_label))
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
    background_superpixels = [i.item() for i in torch.unique(all_superpixels_mask) if i not in relevant_superpixels_thresholded]
    foreground_embeddings = torch.zeros([len(relevant_superpixels_thresholded), 2048])
    background_embeddings = torch.zeros([len(background_superpixels), 2048])
    for i, superpixel in enumerate(relevant_superpixels_thresholded):
        all_superpixels_mask_tmp = all_superpixels_mask.clone()
        all_superpixels_mask_tmp[all_superpixels_mask_tmp != superpixel] = 0
        all_superpixels_mask_tmp[all_superpixels_mask_tmp>0] = 1
        s,l = get_bbox_coordinates_one_box(all_superpixels_mask_tmp)
        base = torch.Tensor(org_img).permute(2,0,1).clone().cpu() / 255
        base_aspm = base.clone()
        base_aspm[0,:,:] = base_aspm[0,:,:] * all_superpixels_mask_tmp
        base_aspm[1,:,:] = base_aspm[1,:,:] * all_superpixels_mask_tmp
        base_aspm[2,:,:] = base_aspm[2,:,:] * all_superpixels_mask_tmp
        cut = base_aspm[:,s[1]:l[1],s[0]:l[0]].unsqueeze(0).to(device)
        with torch.no_grad():
            feat_foreground_sp = model(cut)
            foreground_embeddings[i,:] = feat_foreground_sp
    for i, superpixel in enumerate(background_superpixels):
        all_superpixels_mask_tmp = all_superpixels_mask.clone()
        all_superpixels_mask_tmp[all_superpixels_mask_tmp != superpixel] = 0
        all_superpixels_mask_tmp[all_superpixels_mask_tmp>0] = 1
        s,l = get_bbox_coordinates_one_box(all_superpixels_mask_tmp)
        base = train_input.clone().cpu()
        base_aspm = base.clone()

        base_aspm[0,:,:] = base_aspm[0,:,:] * all_superpixels_mask_tmp
        base_aspm[1,:,:] = base_aspm[1,:,:] * all_superpixels_mask_tmp
        base_aspm[2,:,:] = base_aspm[2,:,:] * all_superpixels_mask_tmp
        cut = base_aspm[:,s[1]:l[1],s[0]:l[0]].unsqueeze(0).to(device)
        with torch.no_grad():
            feat_background_sp = model(cut)
            background_embeddings[i,:] = feat_background_sp
    return foreground_embeddings, background_embeddings, relevant_superpixels_thresholded, all_superpixels_mask


def get_mean_embeddings(data_loader, model, embedding_dir, get_foreground_background_embeddings=get_foreground_background_embeddings, N_SEGMENTS=250, THRESHOLD=0.1,):
    foreground_embeddings = torch.zeros([len(data_loader.dataset), 2048])
    background_embeddings = torch.zeros([len(data_loader.dataset), 2048])
    counter = 0
    for epoch in range(0, 1):
        batch = 0
        with tqdm(data_loader, unit="batch") as tepoch:
            for train_inputs, train_labels, train_org_images in tepoch:
                batch += 1
                tepoch.set_description(f"Epoch {epoch}")
                train_inputs, train_labels, train_org_images = train_inputs.to(DEVICE), train_labels.to(DEVICE), train_org_images.to(DEVICE)
                for i in range(0,train_inputs.shape[0],1):
                    embed_f, embed_b, _, _ = get_foreground_background_embeddings(train_labels[i], train_org_images[i], train_inputs[i], N_SEGMENTS=N_SEGMENTS, threshold=THRESHOLD, model=model)
                    mean_f = torch.mean(embed_f, dim=0)
                    mean_b = torch.mean(embed_b, dim=0)
                    foreground_embeddings[counter,:] += mean_f.cpu()
                    background_embeddings[counter,:] += mean_b.cpu()
                    counter += 1
    overall_foreground_mean = torch.mean(foreground_embeddings, dim=0)
    overall_background_mean = torch.mean(background_embeddings, dim=0)                
    torch.save(overall_foreground_mean, embedding_dir + "foreground.pt")
    torch.save(overall_background_mean, embedding_dir + "background.pt")