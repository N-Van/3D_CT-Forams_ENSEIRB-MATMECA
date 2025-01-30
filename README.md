# 3D-CT-Forams (ENSEIRB-MATMECA - PACEA)

## Overview

This project aims to automate the counting of foraminifera using object detection techniques. We utilize models such as YOLO (You Only Look Once) to detect foraminifera in 2D images obtained from microtomography scans. The project also includes clustering and visualization techniques to refine detection results.

## Project Structure

The directory structure of the project is organized as follows:

```
.
├── datasets
│   ├── LABELME       # Dataset in LabelMe annotation format
│   ├── RAW           # Raw microtomography images (TIF or PNG format)
│   ├── README.md     # Documentation about datasets
│   └── YOLO          # YOLO-specific dataset preparation
│
├── models
│   ├── base          # Pretrained base models for object detection
│   │   ├── SAM       # Segment Anything Model (SAM) weights
│   │   └── YOLO      # YOLO model weights
│   ├── README.md     # Documentation about models
│   └── trained       # Directory containing trained model weights
│       ├── SAM       # Trained SAM models
│       └── YOLO      # Trained YOLO models
│
├── README.md         # Main project documentation
├── requirement.txt   # Required dependencies for the project
│
├── src               # Source code for processing and training
│   ├── clustering    # Clustering algorithms
│   │   ├── hdbscan_clustering.py
│   │   ├── hdbscan_clustering_with_confidence_after.py
│   │   ├── hdbscan_clustering_with_confidence_before.py
│   │   ├── merger.py
│   │   ├── pipeline_compare_confidence.py
│   │   └── README.md
│   │
│   ├── utils         # Utility scripts for data processing
│   │   ├── dataset   # Dataset preparation and transformation tools
│   │   │   ├── 2.5D  # Preparation and transformation tools for 2.5D datasets
│   │   │   ├── apply_bbox_on_dataset.py
│   │   │   ├── erode_dataset.py
│   │   │   ├── split.py
│   │   │   ├── TIF2PNG.py
│   │   │   ├── TIF_CSV_SAM_to_labelme.py
│   │   │   ├── TIF_CSV_SAM_to_PNG.py
│   │   │   ├── TIF_CSV_SAM_to_yolo.py
│   │   │   └── yoloformat2labelme.py
│   │   ├── make_gif.py  # Tool to create GIFs from processed images
│   │   ├── napari       # Napari-based visualization scripts
│   │   │   ├── napari_annotations.py
│   │   │   ├── napari_predictions.py
│   │   │   ├── napari_visualisation_clusters.py
│   │   │   └── README.md
│   │
│   ├── yolo # YOLO-related scripts for training and inference
│   │   ├── 2.5D
│   │   │   └── inference_3D_to_CSV_RGB.py  # YOLO inference in 3 directions and put results in a CSV file with 2.5D images
│   │   ├── inference_3D_to_CSV.py  # YOLO inference in 3 directions and put results in a CSV file
│   │   ├── inference.py  # Run inference with YOLO models
│   │   ├── train.py      # Train YOLO models
│   │   ├── utils        # YOLO-specific utility scripts
│   │   │   ├── convert_to_TensorRT.py  # Convert trained YOLO models to TensorRT for faster inference
│   │   │   ├── yoloinference2labelme.py  # Convert YOLO inference results to LabelMe format
│   │   ├── val.py        # Validate YOLO models on test datasets
│   │   ├── README.md     # Documentation for YOLO training and inference
```

## Getting Started

### 1. Clone the Repository

- With SSH:
```bash
   git clone git@github.com:N-Van/3D-CT-Forams_ENSEIRB-MATMECA.git
```
- With HTTPS:
```bash
   git clone https://github.com/N-Van/3D-CT-Forams_ENSEIRB-MATMECA.git
```

### 2. Set Up the Environment

Ensure you have the required dependencies installed. You can use a virtual environment:

```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirement.txt
```


## Resources

- **Datasets, results and trained models**: [Google Drive Repository](https://drive.google.com/drive/folders/1HA9SrQdXyjDlxhwnYbfOX51E-qSh1Lip?usp=sharing)
- **Slides related to this project**: [Google Slides](https://docs.google.com/presentation/d/1QYHk1Goxds1vmWF4biPpaacvyNd_hNFpewftqWoVnEc/edit?usp=sharing)


## Example

This is an example showcasing the results.
![alt text](output_animation.gif)