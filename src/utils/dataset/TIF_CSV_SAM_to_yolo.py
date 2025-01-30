import os
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
from ultralytics import SAM
from tqdm import tqdm
import tifffile as tiff

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
        int(bounding_box[0] * scale_x),  # a_min
        int(bounding_box[1] * scale_y),  # b_min
        int(bounding_box[2] * scale_x),  # a_max
        int(bounding_box[3] * scale_y)   # b_max
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


def draw_black_masks(image, bounding_boxes):
    for bounding_box in bounding_boxes:
        a_min, b_min, a_max, b_max = bounding_box
        cv2.rectangle(image, (a_min, b_min), (a_max, b_max), color=(0, 0, 0), thickness=-1)
    return image


def save_images_and_annotations(tif_path, csv_path, output_folder, box_width, num_frames, model_path, base_image_name, apply_sam, black_masks, num_frames_to_mask):
    if apply_sam:
        sam_model = SAM(model_path)

    # Create the output folder and necessary subfolders
    os.makedirs(output_folder, exist_ok=True)
    images_folder = os.path.join(output_folder, "images")
    labels_folder = os.path.join(output_folder, "labels")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    # Load the TIFF file
    tif_data = tiff.imread(tif_path)

    # Load CSV data
    df = pd.read_csv(csv_path)

    # For each direcion
    for direction in range(3): # 0: z, 1: y, 2: x

        # Iterate through each image in the TIFF file
        for idx in tqdm( range(tif_data.shape[direction]), desc="Processing images" ):
            
            # Extract the slice along the specified direction
            if direction == 0:  # Along z-axis
                img_array = tif_data[idx, :, :].copy()
            elif direction == 1:  # Along y-axis
                img_array = tif_data[:, idx, :].copy()
            elif direction == 2:  # Along x-axis
                img_array = tif_data[:, :, idx].copy()

            # Convert to BGR format if necessary
            if img_array.ndim == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

            # Gather bounding boxes for the current index and the surrounding frames in that direction
            bounding_boxes = []


            # Get the annotation points for the current image
            if direction == 0:
                points = df[df['z_Foram_Pix'] == idx]
            elif direction == 1:
                points = df[df['y_Foram_Pix'] == idx]
            elif direction == 2:
                points = df[df['x_Foram_Pix'] == idx]
            
            for _, row in points.iterrows():
                if direction == 0:
                    a, b = int(row['x_Foram_Pix']), int(row['y_Foram_Pix'])
                elif direction == 1:
                    a, b = int(row['x_Foram_Pix']), int(row['z_Foram_Pix'])
                elif direction == 2:
                    a, b = int(row['y_Foram_Pix']), int(row['z_Foram_Pix'])
                a_min = max(a - box_width // 2, 0)
                a_max = min(a + box_width // 2, img_array.shape[1])
                b_min = max(b - box_width // 2, 0)
                b_max = min(b + box_width // 2, img_array.shape[0])
                bounding_boxes.append([a_min, b_min, a_max, b_max])
                


            # Initialize a list for storing annotations
            annotations = []
            bounding_boxes_list = []

            # Perform inference for each bounding box
            for bbox in bounding_boxes:
                if not apply_sam:
                    # Calculate YOLO format bounding box
                    x_center = (bbox[0] + bbox[2]) / 2 / img_array.shape[1]
                    y_center = (bbox[1] + bbox[3]) / 2 / img_array.shape[0]
                    width = (bbox[2] - bbox[0]) / img_array.shape[1]
                    height = (bbox[3] - bbox[1]) / img_array.shape[0]
                    annotations.append(f"0 {x_center} {y_center} {width} {height}")
                    bounding_boxes_list = bounding_boxes

                else:
                    mask = apply_sam_model(img_array, bbox, sam_model)
                    if mask is not None:
                        y_indices, x_indices = np.where(mask)  # Get the coordinates of the mask
                        if len(y_indices) > 0 and len(x_indices) > 0:  # Ensure there are coordinates
                            bbox = [
                                np.min(x_indices),  # Left
                                np.min(y_indices),  # Top
                                np.max(x_indices),  # Right
                                np.max(y_indices)   # Bottom
                            ]
                        # Calculate YOLO format bounding box
                        x_center = (bbox[0] + bbox[2]) / 2 / img_array.shape[1]
                        y_center = (bbox[1] + bbox[3]) / 2 / img_array.shape[0]
                        width = (bbox[2] - bbox[0]) / img_array.shape[1]
                        height = (bbox[3] - bbox[1]) / img_array.shape[0]
                        annotations.append(f"0 {x_center} {y_center} {width} {height}")
                        bounding_boxes_list.append(bbox)

            if direction == 0:
                d = "z"
            elif direction == 1:
                d = "y"
            else:
                d = "x"


            # Save the image untouched in the output folder if it has not been processed before
            output_image_path = os.path.join(images_folder, f'{base_image_name}_{d}_{idx}.png')
            if not os.path.exists(output_image_path):
                cv2.imwrite(output_image_path, img_array)

            
            # Write annotations to a file with the same name as the image in the labels folder
            annotation_file_path = os.path.join(labels_folder, f'{base_image_name}_{d}_{idx}.txt')
            with open(annotation_file_path, 'w') as ann_file:
                for ann in annotations:
                    ann_file.write(f"{ann}\n")



            offsets = list(range(-(num_frames // 2), 0)) + list(range(1, num_frames // 2 + 1))
            for offset in offsets:
                idx_to_annotate = idx + offset
                if 0 <= idx_to_annotate < tif_data.shape[direction]:  # Check the boundaries
                    annotation_file_path = os.path.join(labels_folder, f'{base_image_name}_{d}_{idx_to_annotate}.txt')
                    with open(annotation_file_path, 'a') as ann_file:
                        for ann in annotations:
                            ann_file.write(f"{ann}\n")


            # If necessary, draw black masks on the images
            if black_masks == True:        

                offsets = list(range(-(num_frames // 2 + num_frames_to_mask // 2), -(num_frames // 2))) + list(range(num_frames // 2 + 1, num_frames // 2 + num_frames_to_mask // 2 + 1))
        
                for offset in offsets:
                    idx_image_to_process = idx + offset
                    if 0 <= idx_image_to_process < tif_data.shape[direction]:  # Check the boundaries
                        # get the image from the images_folder if already processed before
                        if os.path.exists(os.path.join(images_folder, f'{base_image_name}_{d}_{idx_image_to_process}.png')):
                            img_to_process = cv2.imread(os.path.join(images_folder, f'{base_image_name}_{d}_{idx_image_to_process}.png'))
                            img_to_process = draw_black_masks(img_to_process, bounding_boxes_list)
                            # Réenregistrement de l'image modifiée
                            output_image_path = os.path.join(images_folder, f'{base_image_name}_{d}_{idx_image_to_process}.png')
                            cv2.imwrite(output_image_path, img_to_process)

                        else:
                            # Extract the slice along the specified direction
                            if direction == 0:  # Along z-axis
                                img_to_process = tif_data[idx_image_to_process, :, :].copy()
                            elif direction == 1:  # Along y-axis
                                img_to_process = tif_data[:, idx_image_to_process, :].copy()
                            elif direction == 2:  # Along x-axis
                                img_to_process = tif_data[:, :, idx_image_to_process].copy()
                            # Convert to BGR format if necessary
                            if img_to_process.ndim == 2:  # Grayscale
                                img_to_process = cv2.cvtColor(img_to_process, cv2.COLOR_GRAY2BGR)
                            img_to_process = draw_black_masks(img_to_process, bounding_boxes_list)
                            # Enregistrement de l'image modifiée
                            output_image_path = os.path.join(images_folder, f'{base_image_name}_{d}_{idx_image_to_process}.png')
                            cv2.imwrite(output_image_path, img_to_process)
            




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process TIFF images, apply segmentation using a SAM model, and save images along with LabelMe annotations.')
    parser.add_argument('tif_path', type=str, help='Path to the TIFF file.')
    parser.add_argument('csv_path', type=str, help='Path to the input CSV file with coordinates.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder.')
    parser.add_argument('--box_width', default=20, type=int, help='Width of the bounding box around each point.')
    parser.add_argument('--num_frames', default=5, type=int, help='Total number of frames (above and below in total) to consider.')
    parser.add_argument('--model_path', default="mobile_sam.pt",type=str, help='Path to the SAM model file.')
    parser.add_argument('--base_image_name', default="image", type=str, help='Base name for output images and annotations.')
    parser.add_argument("--apply_sam", action="store_true", help="Use SAM to refine the bbox")
    parser.add_argument("--black_bbox", default=False, type=bool, help="Use to keep some boxes and draw black squares on the others.")
    parser.add_argument('--num_frames_to_mask', default=8, type=int, help='Total number of frames (above and below in total) to consider.')

    args = parser.parse_args()
    

    save_images_and_annotations(args.tif_path, args.csv_path, args.output_folder, args.box_width, args.num_frames, args.model_path, args.base_image_name,args.apply_sam, args.black_bbox, args.num_frames_to_mask)
    print(f'Images and annotations saved in {args.output_folder}.')