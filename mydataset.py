import os
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
from torchvision import transforms
import torch.nn.functional as F


class MHAImageDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, target_size=(128, 128, 128)):
       
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size

        assert len(self.image_paths) == len(self.mask_paths), "Image and mask paths length mismatch."

    def __len__(self):
        return len(self.image_paths)

    def _resize(self, image, target_size):
        """
        Resize 3D image to the target size using trilinear interpolation.
        """
        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, D, H, W]
        resized_image = F.interpolate(image_tensor, size=target_size, mode="trilinear", align_corners=False)
        return resized_image.squeeze().numpy()

    def __getitem__(self, idx):
        
        image = sitk.GetArrayFromImage(sitk.ReadImage(self.image_paths[idx]))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(self.mask_paths[idx]))

        image = self._resize(image.astype(np.float32), self.target_size)
        mask = self._resize(mask.astype(np.float32), self.target_size)

        image = torch.tensor(image).unsqueeze(0) 
        mask = torch.tensor(mask).unsqueeze(0)  

        if self.transform:
            sample = {"image": image, "mask": mask}
            sample = self.transform(sample)
            image, mask = sample["image"], sample["mask"]

        return image, mask

class RandomFlip3D:
    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[2]) 
            mask = torch.flip(mask, dims=[2])
        return {"image": image, "mask": mask}

if __name__ == "__main__":
    import glob
    image_paths = sorted(glob.glob("/home/jingyixu/eel6935/FinalProject/CIS-UNet/data/images/*.mha"))  
    mask_paths  = sorted(glob.glob("/home/jingyixu/eel6935/FinalProject/CIS-UNet/data/masks/*.mha")) 

    transforms_pipeline = transforms.Compose([RandomFlip3D()])
    dataset = MHAImageDataset(image_paths, mask_paths, transform=transforms_pipeline)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    for batch_idx, (images, masks) in enumerate(dataloader):
        print(f"Batch {batch_idx}")
        print(f"Images shape: {images.shape}")  # [B, C, H, W, D]
        print(f"Masks shape: {masks.shape}")    # [B, C, H, W, D]
