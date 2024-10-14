import os
import argparse
import shutil
import random

def split_dataset(input_folder, output_folder, split_ratio):
    # Define input subdirectories for images and labels
    images_folder = os.path.join(input_folder, "images")
    labels_folder = os.path.join(input_folder, "labels")

    # Create output directories
    train_images_dir = os.path.join(output_folder, "images", "train")
    val_images_dir = os.path.join(output_folder, "images", "val")
    train_labels_dir = os.path.join(output_folder, "labels", "train")
    val_labels_dir = os.path.join(output_folder, "labels", "val")

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # List all images in the images folder
    images = [f for f in os.listdir(images_folder) if f.endswith('.png')]
    
    # Shuffle the dataset
    random.shuffle(images)

    # Calculate split index
    split_index = int(len(images) * split_ratio)

    # Split into training and validation sets
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Copy images and labels to the corresponding directories
    for image in train_images:
        shutil.copy(os.path.join(images_folder, image), os.path.join(train_images_dir, image))
        label_file = image.replace('.png', '.txt')
        shutil.copy(os.path.join(labels_folder, label_file), os.path.join(train_labels_dir, label_file))

    for image in val_images:
        shutil.copy(os.path.join(images_folder, image), os.path.join(val_images_dir, image))
        label_file = image.replace('.png', '.txt')
        shutil.copy(os.path.join(labels_folder, label_file), os.path.join(val_labels_dir, label_file))

    print(f'Split dataset into {len(train_images)} training images and {len(val_images)} validation images.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split a dataset into training and validation sets. The dataset should contain "images" and "labels" subfolders. The script will shuffle the dataset and create separate folders for training and validation data based on the specified split ratio.')
    parser.add_argument('input_folder', type=str, help='Path to the input dataset folder containing "images" and "labels" subfolders.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder for the split dataset.')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Ratio of training data (default: 0.8 for 80% train, 20% val).')

    args = parser.parse_args()
    
    split_dataset(args.input_folder, args.output_folder, args.split_ratio)
