import torch
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from mydataset import MHAImageDataset
from monai.metrics import DiceMetric
from utils.CIS_UNet import CIS_UNet
from utils.unet3d import UNet
from utils.transunet import UNetWithTransformer
from utils.vnet3d import VNet3D
import SimpleITK as sitk
import pyvista as pv
pv.start_xvfb()
from monai.transforms import Compose

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetWithTransformer(n_channels=1, 
            n_classes=24, ).to(device)

model.load_state_dict(torch.load("trans3d_mha.pth"))
model.eval()


def inference_and_visualization(image_files, label_files):
    data_set = MHAImageDataset(image_files, label_files, transform=Compose([]))
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False, num_workers=4)
    idx = 0
    for image_tensor, label_tensor in data_loader:

        image_tensor = image_tensor.to(device)  # [1, 1, D, H, W]
        label_tensor = label_tensor.to(device)
        image = image_tensor.cpu().numpy()[0, 0]
        
        with torch.no_grad():
            output = model(image_tensor)

        pred = torch.argmax(output, dim=1, keepdim=True)
        label = label_tensor.cpu().numpy()[0, 0]
        
        print(f"Inference on {idx}:")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        if idx<3:
            axes[0].imshow(image[image.shape[0] // 2, :, :], cmap='viridis')
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            
            axes[1].imshow(label[label.shape[0] // 2, :, :], cmap='viridis')
            axes[1].set_title("Ground Truth Mask")
            axes[1].axis("off")
            
            axes[2].imshow(pred.cpu().numpy()[0, 0, pred.shape[2] // 2, :, :], cmap='viridis')
            axes[2].set_title("Predicted Mask")
            axes[2].axis("off")

            plt.tight_layout()
            os.makedirs("new_vis_3dunet", exist_ok=True)
            plt.savefig(f"new_vis_3dunet/inference_{idx}_slice.png")

            save_3d_image_pyvista(label_tensor.cpu().numpy(), "new_vis_3dunet", f"{idx}_label_3d")
            save_3d_image_pyvista(pred.cpu().numpy(), "new_vis_3dunet", f"{idx}_pred_3d")

        idx += 1



def save_3d_image_pyvista(image_array, save_dir, file_name):
    img_data = image_array[0, 0] 
    
    D, H, W = img_data.shape
    
    grid = pv.StructuredGrid(*np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij'))
    
    grid.point_data["values"] = img_data.flatten(order="F") 

    os.makedirs(save_dir, exist_ok=True)

    plotter = pv.Plotter(off_screen=True)
    plotter.add_volume(grid, cmap="bone", opacity="linear")
    plotter.set_background("white")

    plotter.screenshot(f"{save_dir}/{file_name}_3d.png")
    plotter.close()

image_files = sorted(glob.glob("/home/jingyixu/eel6935/FinalProject/CIS-UNet/data/images/*.mha"))
label_files = sorted(glob.glob("/home/jingyixu/eel6935/FinalProject/data/masks/*.mha"))

inference_and_visualization(image_files, label_files)
