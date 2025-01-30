# Dataset Utility Scripts

This folder contains various utility scripts designed for processing and manipulating datasets, specifically those related to image segmentation, bounding boxes, and file format conversions. 

## Folder Structure

```
src/utils/dataset/
├── 2.5D
├── apply_bbox_on_dataset.py
├── erode_dataset.py
├── README.md
├── split.py
├── TIF2PNG.py
├── TIF_CSV_SAM_to_labelme.py
├── TIF_CSV_SAM_to_PNG.py
├── TIF_CSV_SAM_to_yolo.py
└── yoloformat2labelme.py
```

## Scripts Overview

### 1. **apply_bbox_on_dataset.py**
   - **Description**: This script applies bounding boxes to images in a TIFF file, based on coordinates provided in a CSV file. It processes a given TIFF file, reads the CSV file for coordinates, and draws bounding boxes around the specified points. The processed images are saved as PNGs.
   - **Usage**:  
     ```bash
     python apply_bbox_on_dataset.py <tif_path> <csv_path> <output_folder> <box_width> <num_frames>
     ```
     - `tif_path`: Path to the input TIFF file.
     - `csv_path`: Path to the CSV file containing foraminifera coordinates.
     - `output_folder`: Directory where the processed images will be saved.
     - `box_width`: Width of the bounding box around each point.
     - `num_frames`: Number of frames above and below to consider when drawing bounding boxes.

### 2. **erode_dataset.py**
   - **Description**: This script copies a subset of images and their annotations from an input folder to an output folder, based on a specified frequency. Useful for downsampling datasets.
   - **Usage**:  
     ```bash
     python erode_dataset.py --input_folder <input_folder> --output_folder <output_folder> --frequency <frequency>
     ```
     - `input_folder`: Path to the input folder containing images and annotations.
     - `output_folder`: Path to the output folder where selected files will be copied.
     - `frequency`: Frequency to select images (e.g., select every 10th image).

### 3. **split.py**
   - **Description**: This script splits a dataset into training and validation sets. It assumes the dataset contains two subfolders: "images" and "labels". The script shuffles the dataset and copies the images and corresponding labels into new subfolders for training and validation.
   - **Usage**:  
     ```bash
     python split.py <input_folder> <output_folder> --split_ratio <split_ratio>
     ```
     - `input_folder`: Path to the input dataset folder containing "images" and "labels".
     - `output_folder`: Path to the output folder for the split dataset.
     - `split_ratio`: Ratio of training data (default: 0.8 for 80% train, 20% val).

### 4. **TIF_CSV_SAM_to_labelme.py**
   - **Description**: This script processes saves the results along with LabelMe-style annotations.
   - **Usage**:  
     ```bash
     python TIF_CSV_SAM_to_labelme.py <tif_path> <csv_path> <output_folder> --box_width <box_width> --num_frames <num_frames> --model_path <model_path> --base_image_name <base_image_name> --apply_sam
     ```
     - `tif_path`: Path to the input TIFF file.
     - `csv_path`: Path to the CSV file with foraminifera coordinates.
     - `output_folder`: Path to the output folder where processed images and annotations will be saved.
     - `box_width`: Width of the bounding box around each point.
     - `num_frames`: Number of frames above and below to consider.
     - `model_path`: Path to the SAM model file.
     - `base_image_name`: Base name for output images and annotations.
     - `--apply_sam`: Use SAM model to refine bounding boxes.

### 5. **TIF_CSV_SAM_to_PNG.py**
   - **Description**: Similar to the `TIF_CSV_SAM_to_labelme.py`, but this script saves the processed images as PNG files instead of LabelMe annotations.
   - **Usage**:  
     ```bash
     python TIF_CSV_SAM_to_PNG.py <tif_path> <csv_path> <output_folder> --box_width <box_width> --num_frames <num_frames> --model_path <model_path> --base_image_name <base_image_name> --apply_sam
     ```

### 6. **TIF_CSV_SAM_to_yolo.py**
   - **Description**: This script processes TIFF images and saves the annotations in YOLO format.
   - **Usage**:  
     ```bash
     python TIF_CSV_SAM_to_yolo.py <tif_path> <csv_path> <output_folder> --box_width <box_width> --num_frames <num_frames> --model_path <model_path> --base_image_name <base_image_name> --apply_sam
     ```

### 7. **TIF2PNG.py**
   - **Description**: This script extracts individual frames from a multi-frame TIFF file and saves them as PNG images.
   - **Usage**:  
     ```bash
     python TIF2PNG.py <tif_path> <output_folder>
     ```
     - `tif_path`: Path to the input TIFF file.
     - `output_folder`: Path to the output folder where PNG images will be saved.

### 8. **yoloformat2labelme.py**
   - **Description**: Converts YOLO format annotations to LabelMe format. It reads YOLO annotation files and corresponding images, then outputs LabelMe JSON annotations.
   - **Usage**:  
     ```bash
     python yoloformat2labelme.py <yolo_file> <image_file>
     ```

## Dependencies

These scripts require the following Python libraries:

- `numpy`
- `pandas`
- `opencv-python`
- `tifffile`
- `torch`
- `ultralytics`
- `Pillow`
- `json`
- `shutil`

To install the required dependencies, use the following command:
```bash
pip install -r requirements.txt
```
