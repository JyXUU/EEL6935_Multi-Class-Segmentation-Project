# EEL6935_Multi-Class-Segmentation-Project

# EEL6935 Final Project: Multi-Class Segmentation of the Aorta and Branches

This repository contains the implementation and analysis of state-of-the-art 3D deep learning models for the multi-class segmentation of the aorta and its branches in 3D computed tomography (CT) images. This project is a part of the EEL6935 course at the University of Florida.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methods](#methods)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

## Introduction
Medical image segmentation is crucial for clinical diagnostics and treatment planning. This project evaluates the performance of four advanced 3D segmentation models:
- **3D U-Net**
- **V-Net**
- **TransUNet**
- **CIS-UNet**

The models were trained and tested using the AortaSeg24 dataset, a benchmark for aortic segmentation tasks. The evaluation focuses on both quantitative metrics, such as the Dice coefficient, and qualitative visualizations.

## Dataset
The dataset used is the **AortaSeg24** dataset, which includes high-resolution volumetric CT scans annotated by clinical experts. The dataset is publicly available on the [AortaSeg24 Grand Challenge website](https://aortaseg24.grand-challenge.org/).

### Preprocessing
- Intensity normalization: Values were clipped to \([-175, 250]\) Hounsfield units and scaled to \([0, 1]\).
- Resampling: All volumes were resampled to an isotropic voxel size of \(1.5 \, \text{mm}^3\).
- Foreground cropping: Non-background regions were retained.
- Patch extraction: \(128 \times 128 \times 128\) patches were extracted for training.

## Methods
The following 3D deep learning models were implemented and evaluated:

1. **3D U-Net**: An encoder-decoder architecture with skip connections.
2. **V-Net**: A fully convolutional network with residual connections.
3. **TransUNet**: A hybrid model combining transformers and CNNs.
4. **CIS-UNet**: Incorporates Context-aware Shifted Window Self-Attention for enhanced segmentation.

### Training Setup
- Optimizer: Adam with a learning rate of \(1 \times 10^{-4}\).
- Loss Function: Hybrid Dice-Cross Entropy loss.
- Cross-Validation: 4-fold cross-validation to ensure robust evaluation.
- Data Augmentation: Flips, rotations, intensity shifts, and elastic deformations.

## Results
### Quantitative Evaluation
The models were evaluated using the Dice coefficient. Below are the mean Dice scores:

| Model        | Mean Dice Coefficient |
|--------------|------------------------|
| 3D U-Net     | 0.6297                 |
| V-Net        | 0.6588                 |
| TransUNet    | 0.7024                 |
| CIS-UNet     | 0.7045                 |

### Qualitative Evaluation
Visual comparisons showed that CIS-UNet outperformed the other models in accurately delineating intricate structures of the aorta and its branches.

## Requirements
To replicate this project, the following dependencies are required:
- Python 3.8+
- PyTorch
- MONAI
- NumPy
- Matplotlib
- tqdm

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/JyXUU/EEL6935_Multi-Class-Segmentation-Project.git
   cd EEL6935_Multi-Class-Segmentation-Project
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the dataset: Download the AortaSeg24 dataset and organize it according to the expected directory structure.

4. Train the models:
   ```bash
   python train.py --model <model_name>
   ```

5. Evaluate the models:
   ```bash
   python evaluate.py --model <model_name>
   ```

6. Visualize the results:
   ```bash
   python visualize.py --model <model_name>
   ```

## Citation
If you use this project or dataset, please cite the following sources:

1. AortaSeg24 dataset: [AortaSeg24 Grand Challenge](https://aortaseg24.grand-challenge.org/).
2. Model implementations and additional references are listed in the `References` section of the final report.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
