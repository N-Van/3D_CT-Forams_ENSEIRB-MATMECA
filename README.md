# 3D-CT-Forams (ENSEIRB-MATMECA - PACEA)

## Overview

This project aims to automate the counting of foraminifera by utilizing object detection techniques. Specifically, we will train models such as YOLO (You Only Look Once) to detect foraminifera in 2D images obtained from microtomography scans.

## Project Structure

The directory structure of the project is organized as follows:
```
.
├── datasets
│   ├── LABELME          # Labeling data for the dataset in labelme format
│   ├── RAW              # Raw microtomography images in format TIF or PNG
│   ├── README.md
│   └── YOLO             # YOLO-specific dataset preparation
│       └── example
│           ├── dataset.yaml  # Dataset configuration file
│           ├── images        # Directory containing images
│           └── labels        # Directory containing label files
├── models
│   ├── base              # Base models for object detection
│   │   ├── SAM           # Segment Anything Model weights
│   │   │   ├── mobile_sam.pt
│   │   │   ├── sam_b.pt
│   │   │   └── sam_l.pt
│   │   └── YOLO          # YOLO model weights
│   │       ├── yolo11l.pt
│   │       ├── yolo11n.pt
│   │       ├── yolo11x.pt
│   │       ├── yolov8m.pt
│   │       └── yolov8n.pt
│   ├── README.md
│   └── trained           # Directory for trained model weights
│       └── yolov11n_2048_14102024.pt
├── README.md
├── src                   # Source code for processing and training
│   ├── README.md
│   ├── utils             # Utility scripts for data processing
│   │   ├── apply_bbox_on_dataset.py  # Generate and save PNG images from a TIFF file with bounding boxes drawn around foraminifera coordinates specified in a CSV file.
│   │   ├── TIF2PNG.py    # Extract individual frames from a multi-frame TIFF file and save them as PNG images in the specified output folder.
│   │   └── TIF_CSV_SAM_to_PNG.py  # Process TIFF images to apply segmentation masks and draw bounding boxes around foraminifera coordinates specified in a CSV file. The output is saved as PNG images.
│   └── yolo
│       ├── inference.py   # Script for running inference with YOLO
│       ├── train.py       # Script for training YOLO models
│       └── utils          # YOLO-specific utility scripts
│           ├── cropping_images_dataset.py  # Crop images and update corresponding YOLO annotation files based on the specified crop position and size.
│           ├── dataset_crop_on_bbox.py     # Crop images based on bounding boxes with an optional margin, and adjust YOLO annotations accordingly. The cropped images and updated labels will be saved to the specified output folder.
│           ├── TIF_CSV_SAM_to_yolo.py      # Process TIFF images, apply segmentation using a SAM model, and save images along with YOLO format annotations. The output will include both the processed images and their corresponding label files in a specified output folder.
│           ├── split.py    # Split a dataset into training and validation sets. The dataset should contain "images" and "labels" subfolders. The script will shuffle the dataset and create separate folders for training and validation data based on the specified split ratio.
│           ├── yolopred2labelme.py          # Run inference on a folder of images using a YOLO model and generate LabelMe-compatible JSON annotations.format
│           └── yolotolabelme.py              # Convert YOLO annotations to LabelMe format and save as JSON files along with the images.
└── training              # Training output
```

## Getting Started

1. **Clone the Repository**:

```bash
   git clone git@github.com:N-Van/3D-CT-Forams_ENSEIRB-MATMECA.git
```

2. **Set Up the Environment**:
   Ensure you have the required dependencies installed. You can use a virtual environment:

```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
```
