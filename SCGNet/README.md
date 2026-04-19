# SCGNet

**SCGNet: Semantic Correlation Guided Network for Remote Sensing Semantic Change Detection**

**IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2026**

## Overview

SCGNet is a deep learning framework for **remote sensing semantic change detection**. It captures semantic correlations between bi-temporal images through three novel modules:

- **SGF** (Semantic-Guided Fusion)
- **SCRE** (Semantically Consistent Region Enhancement)
- **MSCA** (Multi-scale Change Activation)

## Requirements

- Python 3.8+
- PyTorch 1.7+
- torchvision
- scikit-image
- numpy
- tensorboardX

## Datasets

- **SECOND**
- **Landsat**

## Training

```bash
# Train on SECOND dataset
python train_SECOND.py

# Train on Landsat dataset
python train_Landsat.py
```

## Inference

```bash
python inference.py
```

