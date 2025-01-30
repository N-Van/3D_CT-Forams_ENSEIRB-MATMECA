import os
import shutil
import argparse
import json
from pathlib import Path

def copy_files(input_folder, output_folder, frequency):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the input folder
    files = sorted(Path(input_folder).glob("*.png"))  # Assuming images are in .jpg format

    for idx, img_file in enumerate(files):
        if idx % frequency == 0:  # Select based on frequency
            # Define the corresponding JSON annotation file (same name as the image)
            json_file = img_file.with_suffix('.json')

            # Copy the image file
            if img_file.exists():
                shutil.copy(img_file, output_folder)
                print(f"Copied image: {img_file.name}")
            
            # Copy the annotation file (if it exists)
            if json_file.exists():
                shutil.copy(json_file, output_folder)
                print(f"Copied annotation: {json_file.name}")
            else:
                print(f"Warning: Annotation file {json_file.name} not found!")

def main():
    parser = argparse.ArgumentParser(description="Copy some images and annotation files from a dataset based on frequency")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing images and annotations")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder to copy the files")
    parser.add_argument("--frequency", type=int, required=True, help="Frequency to copy files (e.g., 10 for every 1 out of 10 images)")

    args = parser.parse_args()

    copy_files(args.input_folder, args.output_folder, args.frequency)

if __name__ == "__main__":
    main()
