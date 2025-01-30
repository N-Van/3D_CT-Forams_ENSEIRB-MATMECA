# Napari-Based Visualization for Foram Detection

## Overview
This module provides visualization tools using **Napari** to explore and analyze foram detection data. It includes scripts for viewing:
- **Annotations** (ground truth foram locations)
- **Predictions** (model-detected foram locations with confidence scores)
- **Clusters** (grouped foram detections after clustering)

These tools allow for an interactive exploration of **3D foram images** stored in TIFF format, along with foram coordinates stored in CSV files.

---
## Folder Architecture
```
src/
└── utils/
    └── napari/
        ├── napari_annotations.py            # Displays ground truth foram annotations
        ├── napari_predictions.py            # Displays foram predictions with confidence
        ├── napari_visualisation_clusters.py # Displays foram clusters after clustering
        ├── README.md                         # This file
```
---
## Installation
Napari is an interactive, multi-dimensional image viewer for Python, well-suited for exploring 3D biological images.

To install Napari and required dependencies, create a new python environnement and run:
```sh
pip install napari tifffile pandas numpy matplotlib scipy
```

---
## Usage
Each script takes a **TIFF image file** and a **CSV file** containing foram coordinates.

### 1. Visualizing Ground Truth Annotations
```sh
python src/utils/napari/napari_annotations.py <tif_file> <annotations.csv> --rectangle True
```
**Arguments:**
- `<tif_file>`: Path to the **TIFF** image stack.
- `<annotations.csv>`: CSV file with ground truth foram locations.
- `--rectangle`: Optional flag to display an annotation boundary.

### 2. Visualizing Model Predictions
```sh
python src/utils/napari/napari_predictions.py <tif_file> <predictions.csv> --rectangle True
```
**Additional Feature:**
- Predictions are colored by **confidence scores** using a colormap.

### 3. Visualizing Clusters
```sh
python src/utils/napari/napari_visualisation_clusters.py <tif_file> <clusters.csv> --barycenters False --rectangle True
```
**Additional Options:**
- `--barycenters True`: Displays only cluster barycenters instead of all points.
- Clusters are colored distinctly, with noise points in **red**.

---
## CSV Format
Each CSV file should be formatted as:
```
id,x_Foram_Pix,y_Foram_Pix,z_Foram_Pix[,conf,cluster]
```
- **Annotations CSV**: Contains `id, x, y, z`.
- **Predictions CSV**: Includes an additional `conf` (confidence score) column.
- **Clusters CSV**: Includes an additional `cluster` column (cluster ID, with `-1` for noise).

