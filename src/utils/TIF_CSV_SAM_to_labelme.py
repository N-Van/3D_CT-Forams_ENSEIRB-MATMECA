import os
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
from ultralytics import SAM
from tqdm import tqdm
import tifffile as tiff
import json

def apply_sam_model(image, bounding_box, sam_model):
    bboxes = torch.tensor([bounding_box])  # Convert to tensor

    # Normalize the image
    image = image.astype(np.float32) / 255.0

    # Ensure the image has three channels
    if image.ndim == 2:  # Grayscale
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]

    # Resize the image to a compatible size (e.g., 1024x1024)
    original_size = image.shape[:2]
    image_resized = cv2.resize(image, (1024, 1024))
    scale_x = 1024 / original_size[1]
    scale_y = 1024 / original_size[0]

    # Adjust bounding box coordinates to match the resized image
    bbox_resized = [
        int(bounding_box[0] * scale_x),  # x_min
        int(bounding_box[1] * scale_y),  # y_min
        int(bounding_box[2] * scale_x),  # x_max
        int(bounding_box[3] * scale_y)   # y_max
    ]

    # Convert to tensor for inference
    image_tensor = torch.tensor(image_resized).permute(2, 0, 1).unsqueeze(0)  # BCHW

    # Perform segmentation using the SAM model
    results = sam_model(image_tensor, bboxes=torch.tensor([bbox_resized]))

    # Collect masks from results
    if results and hasattr(results[0], 'masks'):
        mask = results[0].masks.data  # Shape: (1, height, width)

        # Convert the boolean mask to uint8 (0 or 255)
        mask = mask.cpu().numpy().astype(np.uint8) * 255  # Convert to numpy and scale to 255

        # Resize the mask back to the original image dimensions
        mask_resized = cv2.resize(mask[0], (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        return mask_resized  # Return the resized mask

    return None

def save_images_and_annotations(tif_path, csv_path, output_folder, box_width, num_frames, model_path, base_image_name, apply_sam):
    if apply_sam:
        sam_model = SAM(model_path)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the TIFF file
    tif_data = tiff.imread(tif_path)

    # Load CSV data
    df = pd.read_csv(csv_path)

    # Iterate through each image in the TIFF file
    for z_idx in tqdm(range(tif_data.shape[0]), desc="Processing images"):
        img_array = tif_data[z_idx].copy()  # Get the current image

        # Convert to BGR format if necessary
        if img_array.ndim == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        # Gather bounding boxes for the current z index and the surrounding frames
        bounding_boxes = []
        for offset in range(-num_frames // 2, num_frames // 2 + 1):
            current_z = z_idx + offset
            if 0 <= current_z < tif_data.shape[0]:  # Check bounds
                points = df[df['z_Foram_Pix'] == current_z]
                for _, row in points.iterrows():
                    x, y = int(row['x_Foram_Pix']), int(row['y_Foram_Pix'])
                    x_min = max(x - box_width // 2, 0)
                    x_max = min(x + box_width // 2, img_array.shape[1])
                    y_min = max(y - box_width // 2, 0)
                    y_max = min(y + box_width // 2, img_array.shape[0])
                    bounding_boxes.append([x_min, y_min, x_max, y_max])

        # Initialize a list for storing annotations
        annotations = []

        # Perform inference for each bounding box
        for bbox in bounding_boxes:
            if not apply_sam:
                # Create a rectangle annotation
                annotations.append({
                    "label": "foraminifère",
                    "text": "",
                    "points": [
                        [bbox[0], bbox[1]],  # Top-left corner
                        [bbox[2], bbox[3]]   # Bottom-right corner
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                })
            else:
                mask = apply_sam_model(img_array, bbox, sam_model)
                if mask is not None:
                    # Get the new bounding box from the mask
                    y_indices, x_indices = np.where(mask)  # Get the coordinates of the mask
                    if len(y_indices) > 0 and len(x_indices) > 0:  # Ensure there are coordinates
                        new_bbox = [
                            np.min(x_indices),  # Left
                            np.min(y_indices),  # Top
                            np.max(x_indices),  # Right
                            np.max(y_indices)   # Bottom
                        ]

                        # Create a rectangle annotation with the new bounding box
                        annotations.append({
                            "label": "foraminifère",
                            "text": "",
                            "points": [
                                [new_bbox[0], new_bbox[1]],  # Top-left corner
                                [new_bbox[2], new_bbox[3]]   # Bottom-right corner
                            ],
                            "group_id": None,
                            "shape_type": "rectangle",
                            "flags": {}
                        })

        # Save the image untouched in the output folder
        output_image_path = os.path.join(output_folder, f'{base_image_name}_{z_idx}.png')
        cv2.imwrite(output_image_path, img_array)

        # Create a LabelMe-style annotation file
        annotation_file_path = os.path.join(output_folder, f'{base_image_name}_{z_idx}.json')
        labelme_data = {
            "version": "0.4.10",
            "flags": {},
            "shapes": annotations,
            "imagePath": f'{base_image_name}_{z_idx}.png',
            "imageData": None,
            "imageHeight": img_array.shape[0],
            "imageWidth": img_array.shape[1],
            "text": ""
        }
        with open(annotation_file_path, 'w') as ann_file:
            json.dump(labelme_data, ann_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process TIFF images, apply segmentation using a SAM model, and save images along with LabelMe annotations.')
    parser.add_argument('tif_path', type=str, help='Path to the TIFF file.')
    parser.add_argument('csv_path', type=str, help='Path to the input CSV file with coordinates.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder.')
    parser.add_argument('--box_width', default=20, type=int, help='Width of the bounding box around each point.')
    parser.add_argument('--num_frames', default=5, type=int, help='Number of frames above and below to consider.')
    parser.add_argument('--model_path', default="mobile_sam.pt",type=str, help='Path to the SAM model file.')
    parser.add_argument('--base_image_name', default="image", type=str, help='Base name for output images and annotations.')
    parser.add_argument("--apply_sam", default=True, type=bool, help='Use SAM to refine the bbox')
    

    args = parser.parse_args()
    
    save_images_and_annotations(args.tif_path, args.csv_path, args.output_folder, args.box_width, args.num_frames, args.model_path, args.base_image_name,args.apply_sam)
    print(f'Images and annotations saved in {args.output_folder}.')
