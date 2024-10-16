# YOLO Training and Inference

This repository contains scripts for training a YOLO model and running inference on a set of images using the trained model.


## Training the YOLO Model

To train a YOLO model, use the `train.py` script. This script allows you to specify various parameters, including the dataset, model path, number of epochs, batch size, image size, and project name.

### Command

From the root folder of the project:
```bash
python3 ./src/yolo/train.py <dataset_path> [--model_path <model_path>] [--epochs <epochs>] [--batch_size <batch_size>] [--img_size <img_size>]
```

### Parameters

- `<dataset_path>`: Path to the dataset YAML file containing paths to training and validation images.
- `--model_path`: Path to the YOLO model file (default: `yolo11n.pt`).
- `--epochs`: Number of training epochs (default: `1000`).
- `--batch_size`: Batch size for training (default: `1`).
- `--img_size`: Image size for training (default: `2048`).

### Example

From the root folder of the project:
```bash
python3 ./src/yolo/train.py dataset.yaml --epochs 500 --batch_size 4 --img_size 1024
```

## Running Inference

To run inference on a set of images, use the `inference.py` script. This script processes each image in the specified input folder and saves the output images with bounding boxes to the output folder.

### Command

From the root folder of the project:
```bash
python3 ./src/yolo/inference.py <input_folder> <output_folder> <model_path>
```

### Parameters

- `<input_folder>`: Path to the folder containing input images (supported formats: `.jpg`, `.jpeg`, `.png`).
- `<output_folder>`: Path to the folder where output images will be saved.
- `<model_path>`: Path to the YOLO model file (e.g., `yolo11n.pt`).

### Example

From the root folder of the project:
```bash
python3 ./src/yolo/inference.py input_images/ output_images/ yolo11n.pt
```
