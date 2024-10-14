import os
import argparse
import numpy as np
import pandas as pd
import cv2
import tifffile as tiff
from tqdm import tqdm

def draw_bounding_box(image, bounding_box):
    x_min, y_min, x_max, y_max = bounding_box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
    return image

def save_images_with_bboxes(tif_path, csv_path, output_folder, box_width, num_frames):
    # Load the TIFF file
    tif_data = tiff.imread(tif_path)

    # Load CSV data
    df = pd.read_csv(csv_path)

    # Process each unique z coordinate
    unique_z = df['z_Foram_Pix'].unique()
    
    for z in unique_z:
        # Get the rows corresponding to this z coordinate
        points = df[df['z_Foram_Pix'] == z]
        z_index = int(z)

        # Determine the range of z indices to process
        z_start = max(z_index - num_frames // 2, 0)
        z_end = min(z_index + num_frames // 2 + 1, tif_data.shape[0])

        # Prepare a mask for each image in the range
        for z_idx in range(z_start, z_end):
            img_array = tif_data[z_idx].copy()  # Get the image for this z index

            # Convert to BGR format if necessary
            if img_array.ndim == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

            # Draw bounding boxes for each point in the relevant z slice
            for _, row in points.iterrows():
                x, y = int(row['x_Foram_Pix']), int(row['y_Foram_Pix'])

                # Define the bounding box
                x_min = max(x - box_width // 2, 0)
                x_max = min(x + box_width // 2, img_array.shape[1])
                y_min = max(y - box_width // 2, 0)
                y_max = min(y + box_width // 2, img_array.shape[0])

                # Draw the bounding box on the image
                img_array = draw_bounding_box(img_array, [x_min, y_min, x_max, y_max])

            # Save the output image as PNG
            output_filename = os.path.join(output_folder, f'masked_image_z_{z_idx}.png')
            cv2.imwrite(output_filename, img_array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and save PNG images from a TIFF file with bounding boxes drawn around foraminifera coordinates specified in a CSV file.')
    parser.add_argument('tif_path', type=str, help='Path to the TIFF file.')
    parser.add_argument('csv_path', type=str, help='Path to the input CSV file with coordinates.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder.')
    parser.add_argument('box_width', type=int, help='Width of the bounding box around each point.')
    parser.add_argument('num_frames', type=int, help='Number of frames above and below to consider.')

    args = parser.parse_args()
    
    save_images_with_bboxes(args.tif_path, args.csv_path, args.output_folder, args.box_width, args.num_frames)
    print(f'Images with bounding boxes saved to {args.output_folder}')
