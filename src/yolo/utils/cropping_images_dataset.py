import os
import argparse
import cv2

def crop_images_and_update_labels(input_dir, output_dir, crop_position, crop_size):
    # Define input and output directories for images and labels
    input_images_dir = os.path.join(input_dir, "images")
    input_labels_dir = os.path.join(input_dir, "labels")
    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")

    # Create output directories
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    crop_x, crop_y = crop_position
    crop_width, crop_height = crop_size

    # Process each image and corresponding label
    for img_file in os.listdir(input_images_dir):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            # Read the image
            img_path = os.path.join(input_images_dir, img_file)
            image = cv2.imread(img_path)

            # Crop the image
            cropped_image = image[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
            cv2.imwrite(os.path.join(output_images_dir, img_file), cropped_image)

            # Update the corresponding label file
            label_file = img_file.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
            label_path = os.path.join(input_labels_dir, label_file)

            updated_labels = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue  # Skip empty lines
                        class_id = parts[0]  # Assuming class id is the first part
                        bbox = list(map(float, parts[1:5]))  # Assuming bbox format: x_center, y_center, width, height

                        # Convert to absolute coordinates
                        x_center = bbox[0] * image.shape[1]
                        y_center = bbox[1] * image.shape[0]
                        width = bbox[2] * image.shape[1]
                        height = bbox[3] * image.shape[0]

                        # Check if the bounding box is within the crop
                        if (crop_x <= x_center <= crop_x + crop_width) and (crop_y <= y_center <= crop_y + crop_height):
                            # Adjust bounding box coordinates to the cropped image
                            x_center_new = (x_center - crop_x) / crop_width
                            y_center_new = (y_center - crop_y) / crop_height
                            width_new = width / crop_width
                            height_new = height / crop_height

                            updated_labels.append(f"{class_id} {x_center_new} {y_center_new} {width_new} {height_new}")

            # Write updated labels to new file (or empty if no valid labels)
            with open(os.path.join(output_labels_dir, label_file), 'w') as f:
                if updated_labels:
                    f.write("\n".join(updated_labels))
                # If updated_labels is empty, it will save an empty file

def main():
    parser = argparse.ArgumentParser(description='Crop images and update corresponding YOLO annotation files based on the specified crop position and size.')
    parser.add_argument('input_dir', type=str, help='Path to the input directory containing "images" and "labels" subfolders.')
    parser.add_argument('output_dir', type=str, help='Path to the output directory for cropped images and updated labels.')
    parser.add_argument('--crop_position', type=int, nargs=2, default=(362, 362), help='Top-left corner of the crop (x, y).')
    parser.add_argument('--crop_size', type=int, nargs=2, default=(1176, 1176), help='Width and height of the crop (width, height).')

    args = parser.parse_args()

    crop_images_and_update_labels(args.input_dir, args.output_dir, tuple(args.crop_position), tuple(args.crop_size))

    print('Cropping completed!')

if __name__ == "__main__":
    main()
