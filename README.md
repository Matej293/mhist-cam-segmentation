# Evaluating Semantic Segmentation Performance Using DeepLabv3+ with Pretrained ResNet Backbones and Pseudo-Masks from Class Activation Maps

## Description

This project aims to implement a weakly supervised segmentation pipeline on the Minimalist Histopathology Image Analysis dataset (*MHIST*). It first trains a ResNet34 classifier to generate Class Activation Maps (CAMs), refines them via morphology and DenseCRF to produce pseudo-masks, then trains a DeepLabV3+ model on those masks. The full workflow includes data loading, augmentation, model training (classification and segmentation), pseudo-mask extraction, evaluation, and comprehensive WandB-logged visualizations (error maps, uncertainty maps, detailed overlays).

## Features

- **Classification Pre-training** with ResNet34, BCE+Focal-Tversky loss, one-cycle LR and progressive layer unfreezing  
- **CAM-based Pseudo-Mask Generation** using SmoothGradCAM++, thresholding, morphology (open/close), small-object removal, DenseCRF refinement  
- **Segmentation Training** with DeepLabV3+ (binary Dice + Lovász loss), slide-level classification head, progressive introduction of negative (HP) samples using a set linear probability gradient
- **Data Augmentation** via Albumentations for both classification and segmentation tasks  
- **Experiment Tracking** with WandB: scalar metrics, basic & detailed visualizations, error & uncertainty maps  
- **Utility Modules** for tissue masking, stain normalization (Macenko), connected-component filtering  

## Prerequisites

- Python ≥ 3.6  
- CUDA-enabled GPU  
- Install dependencies:
  ```bash
  pip install -r requirements.txt   
- Install latest pydensecrf:
  ```bash
  pip install git+https://github.com/lucasb-eyer/pydensecrf.git
- Dataset:  https://bmirds.github.io/MHIST/
## Usage

1. **Clone the repo**  
   git clone https://github.com/Matej293/mhist-cam-segmentation.git && cd mhist-cam-segmentation

2. **Prepare data**  
   - Place MHIST images in `data/images/`  
   - Place annotation CSV at `data/mhist_annotations.csv`  

3. **Configure**  
   Edit `config/default_config.yaml` to adjust data paths, hyperparameters, augmentations, and WandB project name

4. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   pip install git+https://github.com/lucasb-eyer/pydensecrf.git
5. **Login to Wandb and run the full pipeline**  
   ```bash
   python main.py
6. **Inspect results**  
   - Classification weights: `models/classifier_best.pth`  
   - Pseudo-masks: `data/pseudo_masks/`  
   - Segmentation weights: `models/segmentation_best.pth`
   - Visualizations: logged on WandB and saved in `data/visualizations/` if `export_detailed` is set to `True` in `main.py`
