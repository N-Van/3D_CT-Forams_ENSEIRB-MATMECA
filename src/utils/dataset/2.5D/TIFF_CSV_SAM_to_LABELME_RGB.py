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

def create_rgb_image(tif_data, index):
    """
    Create an RGB image using three consecutive frames from the TIFF stack.
    """
    r_channel = tif_data[index - 1] if index > 0 else tif_data[index]
    g_channel = tif_data[index]
    b_channel = tif_data[index + 1] if index < tif_data.shape[0] - 1 else tif_data[index]
    
    return np.stack((r_channel, g_channel, b_channel), axis=-1)

def apply_sam_model(image, bounding_box, sam_model):
    bboxes = torch.tensor([bounding_box])  # Convert to tensor

    # Normalize the image
    image = image.astype(np.float32) / 255.0

    # Resize the image to a compatible size (e.g., 1024x1024)
    original_size = image.shape[:2]
    image_resized = cv2.resize(image, (1024, 1024))
    scale_x = 1024 / original_size[1]
    scale_y = 1024 / original_size[0]

    # Adjust bounding box coordinates to match the resized image
    bbox_resized = [
        int(bounding_box[0] * scale_x),  
        int(bounding_box[1] * scale_y),  
        int(bounding_box[2] * scale_x),  
        int(bounding_box[3] * scale_y)   
    ]

    # Convert to tensor for inference
    image_tensor = torch.tensor(image_resized).permute(2, 0, 1).unsqueeze(0)  # BCHW

    # Perform segmentation using the SAM model
    results = sam_model(image_tensor, bboxes=torch.tensor([bbox_resized]))

    if results and hasattr(results[0], 'masks'):
        mask = results[0].masks.data  
        mask = mask.cpu().numpy().astype(np.uint8) * 255 
        mask_resized = cv2.resize(mask[0], (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        return mask_resized  

    return None

def save_images_and_annotations(tif_path, csv_path, output_folder, box_width, num_frames, model_path, base_image_name, apply_sam):
    if apply_sam:
        sam_model = SAM(model_path)

    os.makedirs(output_folder, exist_ok=True)
    tif_data = tiff.imread(tif_path)
    df = pd.read_csv(csv_path)

    for z_idx in tqdm(range(tif_data.shape[0]), desc="Processing images"):
        rgb_image = create_rgb_image(tif_data, z_idx)

        bounding_boxes = []
        for offset in range(-num_frames // 2, num_frames // 2 + 1):
            current_z = z_idx + offset
            if 0 <= current_z < tif_data.shape[0]:  
                points = df[df['z_Foram_Pix'] == current_z]
                for _, row in points.iterrows():
                    x, y = int(row['x_Foram_Pix']), int(row['y_Foram_Pix'])
                    x_min = max(x - box_width // 2, 0)
                    x_max = min(x + box_width // 2, rgb_image.shape[1])
                    y_min = max(y - box_width // 2, 0)
                    y_max = min(y + box_width // 2, rgb_image.shape[0])
                    bounding_boxes.append([x_min, y_min, x_max, y_max])

        annotations = []

        for bbox in bounding_boxes:
            if not apply_sam:
                annotations.append({
                    "label": "foraminifère",
                    "text": "",
                    "points": [
                        [bbox[0], bbox[1]],  
                        [bbox[2], bbox[3]]   
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                })
            else:
                mask = apply_sam_model(rgb_image, bbox, sam_model)
                if mask is not None:
                    y_indices, x_indices = np.where(mask)
                    if len(y_indices) > 0 and len(x_indices) > 0:  
                        bbox = [
                            np.min(x_indices),  
                            np.min(y_indices),  
                            np.max(x_indices),  
                            np.max(y_indices)   
                        ]
                        annotations.append({
                            "label": "foraminifère",
                            "text": "",
                            "points": [
                                [int(bbox[0]), int(bbox[1])],  
                                [int(bbox[2]), int(bbox[3])]   
                            ],
                            "group_id": None,
                            "shape_type": "rectangle",
                            "flags": {}
                        })

        output_image_path = os.path.join(output_folder, f'{base_image_name}_{z_idx}.png')
        cv2.imwrite(output_image_path, rgb_image)

        annotation_file_path = os.path.join(output_folder, f'{base_image_name}_{z_idx}.json')
        labelme_data = {
            "version": "0.4.10",
            "flags": {},
            "shapes": annotations,
            "imagePath": f'{base_image_name}_{z_idx}.png',
            "imageData": None,
            "imageHeight": rgb_image.shape[0],
            "imageWidth": rgb_image.shape[1],
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
    parser.add_argument("--apply_sam", action="store_true", help="Use SAM to refine the bbox")

    args = parser.parse_args()
    
    save_images_and_annotations(args.tif_path, args.csv_path, args.output_folder, args.box_width, args.num_frames, args.model_path, args.base_image_name, args.apply_sam)
    print(f'Images and annotations saved in {args.output_folder}.')
