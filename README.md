# Semantic-Segmentation

This project explores the comparative effectiveness of three prominent architectures: FCN, PSPNet, and U-Net. By implementing and evaluating these models, I gain valuable insights into their strengths and limitations. Augmentation techniques introduce diverse image manipulations, enhancing our understanding of segmentation challenges. Hyperparameter optimization plays a pivotal role in shaping model performance, allowing for significant accuracy improvements. Evaluation metrics provide insights for model comparison and identification of areas for improvement. It implements a semantic segmentation pipeline using PyTorch Lightning.

## Overview

Semantic segmentation is the task of assigning a class label to each pixel in an image. In this project, we provide a full end-to-end pipeline that:

- **Reads and processes image data** from a custom dataset (in our case, a version of the Cam101 dataset).
- **Implements data augmentation** and preprocessing to improve model robustness.
- **Creates a custom PyTorch Dataset** for handling images and corresponding segmentation masks.
- **Defines three segmentation architectures:**
  - **UNet with a ResNet34 backbone**
  - **FPN (FCN-like) with an inceptionresnetv2 backbone**
  - **PSPNet with a ResNet50 backbone**
- **Utilizes PyTorch Lightning** to simplify training, testing, and evaluation routines.
- **Plots training metrics** such as loss, mean Intersection over Union (mIoU), and pixel accuracy.
- **Evaluates the models** on a test set and displays predictions alongside ground truth.

---

## Installation

This project requires Python 3.7+ and the following dependencies:

- torch
- torchvision
- pytorch_lightning
- segmentation-models-pytorch
- torchmetrics
- albumentations
- opencv-python
- numpy
- Pillow
- certifi

## Dataset

The project uses custom datasets where each sample consists of an image and its corresponding segmentation mask. The `label_colors.txt` file maps RGB values to class labels, enabling conversion of the colormap in segmentation masks to label indices.

Make sure to set the correct paths within the code (e.g., `train_path` and `test_path` in `dataset.py`) to point to your dataset location.

---

## Usage

### Data Preparation

1. **Reading Images:**  
   The provided `read_cam_images` function reads images and segmentation masks from the specified file path. It expects file names to follow a consistent naming convention (e.g., image files ending with `.png` and corresponding masks with an appended `_L.png`).

2. **Custom Dataset & Augmentation:**  
   The custom `ImageDataset` class handles image loading, color space conversion, and applies augmentations (using Albumentations) and preprocessing (normalization and conversion to PyTorch tensors).

3. **Data Loaders:**  
   Training and validation loaders are created by splitting the dataset, and a separate test loader is used for evaluation.

### Model Training & Evaluation

The project defines a PyTorch Lightning module (`SegmentModel`) to encapsulate the model, loss functions, and training/validation steps. To train a model:

1. **Select an Architecture:**  
   For example, instantiate a UNet model with a ResNet34 backbone using:
   ```python
   import segmentation_models_pytorch as smp
   UNet_resnet34 = smp.Unet(
       encoder_name="resnet34",
       encoder_weights="imagenet",
       in_channels=3,
       classes=n_classes
   )
   model = SegmentModel(UNet_resnet34)
   ```

2. **Set Up the Trainer:**  
   Use PyTorch Lightning's `Trainer` class:
   ```python
   from pytorch_lightning import Trainer
   trainer = Trainer(max_epochs=20, accelerator="cuda")
   trainer.fit(model)
   ```

3. **Evaluate on Test Data:**  
   After training, test the model using:
   ```python
   trainer.test(dataloaders=test_loader)
   ```

4. **Visualization:**  
   Utility functions (in `utils.py`) are provided to plot training/validation metrics and to visualize predictions against original labels.

---

## Results

### UNet Model (ResNet34 Backbone)
- **Training Time:** Approximately 7.25 minutes
- **Evaluation Metrics (Test Set):**
  - Test Accuracy: ~85.17%
  - Test mIoU: ~0.1736
  - Test Loss: ~0.5147

### FCN/FPN Model (InceptionResNetV2 Backbone)
- **Training Time:** Approximately 3.92 minutes
- **Evaluation:** Similar testing procedures as above with metric plots showing training/validation losses, mIoU, and pixel accuracy.

### PSPNet (ResNet50 Backbone)
- **Training Time:** Approximately 6.05 minutes
- **Evaluation Metrics (Test Set):**
  - Test Accuracy: ~68.41%
  - Test mIoU: ~0.1258
  - Test Loss: ~0.4660

The repository includes scripts and functions to generate detailed metric plots (loss, mIoU, and pixel accuracy over epochs) and to visually compare predictions with ground truth.

---

## Model Architectures & Future Improvements

Current models implemented:
- **UNet (ResNet34 Backbone):** Effective for datasets with moderate complexity.
- **FCN/FPN (InceptionResNetV2 Backbone):** Provides a balance between learning capacity and computational efficiency.
- **PSPNet (ResNet50 Backbone):** Explores pyramid spatial pooling to capture multi-scale context.

**Future enhancements could include:**
- Adding more backbone options (e.g., EfficientNet).
- Experimenting with advanced loss functions (e.g., combined Dice and focal losses).
- Optimizing hyperparameters further with automated tools.
- Incorporating post-processing techniques to refine segmentation outputs.

