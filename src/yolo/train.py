import os
import yaml
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr
from ultralytics.data.augment import Albumentations

def __init__(self, p=1.0):
    """Initialize the transform object for YOLO bbox formatted params."""
    self.p = p
    self.transform = None
    self.contains_spatial = True
    prefix = colorstr("albumentations: ")
    try:
        import albumentations as A

        # Define a series of image augmentation transformations
        T = [
            A.MedianBlur(p=0.01),
            A.CLAHE(p=0.01),
            A.RandomGamma(p=0.01),
            A.ImageCompression(quality_lower=75, p=0.25),
            A.RandomRotate90(p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.BBoxSafeRandomCrop(erosion_rate=0.2, p=1.0),
        ]  # List of augmentations
        # Combine transformations with bounding box parameters
        self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
    except ImportError:  # If the package is not installed, skip
        pass
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")

# Override the __init__ method of the Albumentations class
Albumentations.__init__ = __init__

# Training function
def train_yolo(dataset_path, model_path, epochs, batch_size, img_size, project_name):
    """Train the YOLO model with the specified dataset and parameters."""
    
    # Initialize YOLO model with the given model path
    model = YOLO(model_path)  # Load the specified YOLO model
    Albumentations.__init__ = __init__
    
    # Train the model with the augmented dataset
    model.train(data=dataset_path,
                epochs=epochs, batch=batch_size, imgsz=img_size,
                project=project_name, augment=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLO model with augmentations.')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset YAML file.')
    parser.add_argument('--model_path', type=str, default='yolo11n.pt', help='Path to the YOLO model file (default: yolo11n.pt).')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs (default: 1000).')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training (default: 1).')
    parser.add_argument('--img_size', type=int, default=2048, help='Image size for training (default: 2048).')
    parser.add_argument('--project_name', type=str, default='yolov8_training', help='Name of the project (default: yolov8_training).')
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Call the training function with parsed arguments
    train_yolo(args.dataset_path, args.model_path, args.epochs, args.batch_size, args.img_size, args.project_name)
