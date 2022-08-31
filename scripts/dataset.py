from torch.utils.data import Dataset
class Colonoscopy_Dataset(Dataset):
    def __init__(self, X, Y, img_transform=img_transform, limit_dataset_size=None):
        self.X = X
        self.Y = Y
        self.img_transform = img_transform
        self.limit_dataset_size = limit_dataset_size
    def __len__(self):
      if self.limit_dataset_size is not None: 
        return self.limit_dataset_size
      else:
        return len(self.X)
    
    def __getitem__(self, index):
        # load image
        img = imread(self.X[index])
        # load mask
        if ".tif" in self.Y[index]:
          mask = torch.tensor(imread(self.Y[index])).long()
        elif ".png" in self.Y[index]:
          mask = torch.Tensor(np.array(Image.open(self.Y[index]))).long()
        mask[mask>0] = 1
        img_org = img
        img_transformed = self.img_transform(img)
        return img_transformed, mask, img_org