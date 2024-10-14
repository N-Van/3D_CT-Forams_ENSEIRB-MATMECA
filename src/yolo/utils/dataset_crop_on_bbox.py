import os
import argparse
import cv2

def crop_images_with_bboxes(input_folder, output_folder, margin):
    # Define input subdirectories for images and labels
    train_images_folder = os.path.join(input_folder, "images", "train")
    val_images_folder = os.path.join(input_folder, "images", "val")
    train_labels_folder = os.path.join(input_folder, "labels", "train")
    val_labels_folder = os.path.join(input_folder, "labels", "val")

    # Create output directories for cropped images and labels
    cropped_train_images_dir = os.path.join(output_folder, "images", "train")
    cropped_val_images_dir = os.path.join(output_folder, "images", "val")
    cropped_train_labels_dir = os.path.join(output_folder, "labels", "train")
    cropped_val_labels_dir = os.path.join(output_folder, "labels", "val")
    
    os.makedirs(cropped_train_images_dir, exist_ok=True)
    os.makedirs(cropped_val_images_dir, exist_ok=True)
    os.makedirs(cropped_train_labels_dir, exist_ok=True)
    os.makedirs(cropped_val_labels_dir, exist_ok=True)

    # Function to process a folder of images and labels
    def process_folder(images_folder, labels_folder, cropped_images_dir, cropped_labels_dir):
        images = [f for f in os.listdir(images_folder) if f.endswith('.png')]
        
        for image_name in images:
            # Load image
            image_path = os.path.join(images_folder, image_name)
            image = cv2.imread(image_path)

            # Load corresponding label file
            label_file = image_name.replace('.png', '.txt')
            label_path = os.path.join(labels_folder, label_file)

            if not os.path.exists(label_path):
                print(f"Warning: Label file {label_file} does not exist for {image_name}. Skipping.")
                continue

            # Read bounding boxes from label file
            with open(label_path, 'r') as file:
                bboxes = []
                for line in file:
                    # Assuming YOLO format: class_id x_center y_center width height (normalized)
                    bboxes.append(list(map(float, line.strip().split())))

                # Create a separate cropped image for each bounding box
                for bbox in bboxes:
                    class_id, x_center, y_center, width, height = bbox

                    # Convert to pixel values
                    img_height, img_width, _ = image.shape
                    x_center_pixel = int(x_center * img_width)
                    y_center_pixel = int(y_center * img_height)
                    width_pixel = int(width * img_width)
                    height_pixel = int(height * img_height)

                    # Calculate bounding box coordinates with margin
                    x_min = int(x_center_pixel - width_pixel / 2) - margin
                    y_min = int(y_center_pixel - height_pixel / 2) - margin
                    x_max = int(x_center_pixel + width_pixel / 2) + margin
                    y_max = int(y_center_pixel + height_pixel / 2) + margin

                    # Ensure the bounding box is within the image bounds
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(img_width, x_max)
                    y_max = min(img_height, y_max)

                    # Crop the image
                    cropped_image = image[y_min:y_max, x_min:x_max]

                    # Generate a unique name for the cropped image
                    cropped_image_name = f"{os.path.splitext(image_name)[0]}_class{int(class_id)}_{len(os.listdir(cropped_images_dir))}.png"
                    cropped_image_path = os.path.join(cropped_images_dir, cropped_image_name)
                    cv2.imwrite(cropped_image_path, cropped_image)

                    # Calculate new bounding box coordinates in the cropped image
                    new_x_center = (x_center_pixel - x_min) / (x_max - x_min)
                    new_y_center = (y_center_pixel - y_min) / (y_max - y_min)
                    new_width = width_pixel / (x_max - x_min)
                    new_height = height_pixel / (y_max - y_min)

                    # Save the new annotation for this cropped image
                    cropped_label_path = os.path.join(cropped_labels_dir, cropped_image_name.replace('.png', '.txt'))
                    with open(cropped_label_path, 'w') as f:
                        f.write(f"{int(class_id)} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}")

    # Process both training and validation folders
    process_folder(train_images_folder, train_labels_folder, cropped_train_images_dir, cropped_train_labels_dir)
    process_folder(val_images_folder, val_labels_folder, cropped_val_images_dir, cropped_val_labels_dir)

    print(f'Cropped images saved to {output_folder}/cropped_images and labels saved to {output_folder}/cropped_labels.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop images based on bounding boxes with an optional margin, and adjust YOLO annotations accordingly. The cropped images and updated labels will be saved to the specified output folder.')
    parser.add_argument('input_folder', type=str, help='Path to the input dataset folder containing "images" and "labels" subfolders.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder for cropped images and labels.')
    parser.add_argument('--margin', type=int, default=0, help='Margin to add around the bounding box (default: 0).')

    args = parser.parse_args()

    crop_images_with_bboxes(args.input_folder, args.output_folder, args.margin)
