# Few-Shot-Semantic-Segmentation

This repository contains a course project on **semantic segmentation of land-cover classes** in the **OpenEarthMap** dataset under a **few-shot / limited-data** setting.

The goal is to predict a land-cover label (building, road, vegetation, water, etc.) for every pixel in a high-resolution satellite image using only a small number of labeled examples.

---

## Overview

- **Task:** Land-cover semantic segmentation (pixel-wise classification)  
- **Main model:** DeepLabV3 (ResNet backbone)  
- **Other models:** FCN-ResNet50, U-Net, and a custom PSPNet-style model  
- **Framework:** PyTorch / Torchvision  

The code here is a cleaned-up subset of my course work, focused on the parts I implemented myself.