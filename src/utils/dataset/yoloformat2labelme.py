import os
import json
import argparse
from PIL import Image
import shutil

def yolo_to_labelme(yolo_file, image_file):
    labelme_format = {
        "version": "4.5.6",  
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(image_file),
        "imageData": None,
    }

    # Get image dimensions
    with Image.open(image_file) as img:
        width, height = img.size
        labelme_format["imageHeight"] = height
        labelme_format["imageWidth"] = width

    # Read YOLO file
    if os.path.getsize(yolo_file) == 0:  # Check if the file is empty
        return labelme_format  # Return with empty shapes

    with open(yolo_file, 'r') as f:
        for line in f.readlines():
            class_id, x_center_norm, y_center_norm, width_norm, height_norm = map(float, line.strip().split())
            if class_id != 0:  # Skip if not class 0
                continue
            
            # Convert normalized YOLO coordinates to pixel values
            x_center = x_center_norm * width
            y_center = y_center_norm * height
            width_px = width_norm * width
            height_px = height_norm * height
            
            # Calculate corner points
            top_left_x = max(0, x_center - width_px / 2)
            top_left_y = max(0, y_center - height_px / 2)
            bottom_right_x = min(width, x_center + width_px / 2)
            bottom_right_y = min(height, y_center + height_px / 2)

            # Convert YOLO format to LabelMe format (using two points)
            labelme_shape = {
                "label": "foraminifere",
                "points": [
                    [top_left_x, top_left_y],
                    [bottom_right_x, bottom_right_y]
                ],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            labelme_format["shapes"].append(labelme_shape)

    return labelme_format

def convert_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_dir = os.path.join(input_dir, 'images')
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):  
            base_name = os.path.splitext(filename)[0]
            yolo_file = os.path.join(input_dir, "labels", f'{base_name}.txt')
            image_file = os.path.join(image_dir, filename)

            labelme_data = yolo_to_labelme(yolo_file, image_file)

            # Create the JSON filename
            json_file = os.path.join(output_dir, f'{base_name}.json')

            with open(json_file, 'w') as jf:
                json.dump(labelme_data, jf, indent=4)
            print(f'Converted {yolo_file} to {json_file}')

            # Copy the image to the output directory
            shutil.copy(image_file, os.path.join(output_dir, filename))
            print(f'Copied {image_file} to {os.path.join(output_dir, filename)}')

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO annotations to LabelMe format and save as JSON files along with the images.')
    parser.add_argument('--input_dir', required=True, help='Directory containing YOLO annotation files and images.')
    parser.add_argument('--output_dir', required=True, help='Directory to save LabelMe JSON files and images.')
    
    args = parser.parse_args()
    
    convert_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
