Here is the complete README with all the details from your provided code:

---

# YOLO Inference and Training

This directory contains scripts for training, validating, and running inference using YOLO models on both 2D and 3D datasets.

## Folder Structure
```
src/yolo/
├── 2.5D
│   └── inference_3D_to_CSV_RGB.py
├── inference_3D_to_CSV.py
├── inference.py
├── README.md
├── train.py
├── utils
│   ├── convert_to_TensorRT.py
│   └── yoloinference2labelme.py
└── val.py
```

## Installation

Ensure you have Python installed and the required dependencies:
```bash
pip install ultralytics opencv-python numpy tifffile codecarbon albumentations
```

## Usage

### Training the YOLO Model
To train the YOLO model with augmentation and track energy consumption:
```bash
python train.py <dataset_yaml> --model_path <model_file> --epochs <num_epochs> --batch_size <batch_size> --img_size <image_size>
```
Example:
```bash
python train.py data.yaml --model_path yolov11n.pt --epochs 1000 --batch_size 2 --img_size 1280
```

### Validation
To validate a trained YOLO model:
```bash
python val.py --model <model_file> --data <dataset_yaml> --output <output_csv>
```
Example:
```bash
python val.py --model best.pt --data data.yaml --output validation_results.csv
```

### Inference on 2D Images
To run inference on a folder of images:
```bash
python inference.py <input_folder> <output_folder> <model_file>
```
Example:
```bash
python inference.py images/ output/ yolov11n.pt
```

### Inference on 3D Images
To run inference on a 3D `.tif` file and output results as a CSV:
```bash
python inference_3D_to_CSV.py <input_file.tif> <output_file.csv> <model_file> --conf <confidence_threshold>
```
Example:
```bash
python inference_3D_to_CSV.py volume.tif output.csv yolov11n.pt --conf 0.3
```

### Inference on 3D Images to RGB with CSV Output
To run inference on a 3D `.tif` file, generate synthetic RGB slices, and output results to a CSV:
```bash
python inference_3D_to_CSV_RGB.py <input_file.tif> <output_csv> <model_file> <rgb_size> --conf <confidence_threshold>
```
Example:
```bash
python inference_3D_to_CSV_RGB.py volume.tif output.csv yolov11n.pt 3 --conf 0.25
```

### YOLO Inference to LabelMe Annotations
To run inference on a folder of images using a YOLO model and generate LabelMe-compatible JSON annotations:
```bash
python yoloinference2labelme.py <input_folder> <output_folder> <model_file> --conf <confidence_threshold>
```
Example:
```bash
python yoloinference2labelme.py images/ output/ yolov11n.pt --conf 0.3
```

## Utilities

- **convert_to_TensorRT.py**: Converts YOLO models to TensorRT format for optimized inference on compatible hardware.
  - Usage example:
    ```bash
    python convert_to_TensorRT.py
    ```

- **yoloinference2labelme.py**: Converts YOLO inference results into LabelMe annotations (JSON format).
  - This script processes images in a folder, runs inference using the YOLO model, and saves the results in LabelMe-compatible JSON files.
  - Example usage:
    ```bash
    python yoloinference2labelme.py <input_folder> <output_folder> <model_file> --conf <confidence_threshold>
    ```

## Notes

- Ensure the dataset YAML file is correctly formatted before training or validation.
- The confidence threshold for inference can be adjusted using the `--conf` parameter.
- Energy consumption tracking is included using `CodeCarbon` in training and 3D inference scripts.
  - For the `inference_3D_to_CSV_RGB.py` script, energy consumption tracking is enabled by `CodeCarbon` during the inference process and results in detailed energy and emission reports.

## Example Workflow

1. Train the YOLO model on a custom dataset:
   ```bash
   python train.py data.yaml --model_path yolov11n.pt --epochs 1000 --batch_size 2 --img_size 1280
   ```
   
2. Validate the model on a test dataset:
   ```bash
   python val.py --model best.pt --data data.yaml --output validation_results.csv
   ```

3. Run inference on 2D images and save the results:
   ```bash
   python inference.py images/ output/ yolov11n.pt
   ```

3. Run inference on a 3D `.tif` file and save the results:
   ```bash
   python inference_3D_to_CSV.py volume.tif output.csv yolov11n.pt --conf 0.3
   ```

3. Generate synthetic RGB images from a 3D `.tif` stack and save the results:
   ```bash
   python inference_3D_to_CSV_RGB.py volume.tif output.csv yolov11n.pt 3 --conf 0.25
   ```
