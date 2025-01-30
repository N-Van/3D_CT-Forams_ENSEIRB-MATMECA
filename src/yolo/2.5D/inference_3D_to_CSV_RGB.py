import os
import argparse
import csv
import numpy as np
import tifffile as tiff
import cv2
from ultralytics import YOLO
from codecarbon import EmissionsTracker

def generate_rgb_images(img_stack, direction, rgb_size):
    """
    Generate synthetic RGB images from grayscale 3D image stacks.

    Args:
        img_stack: 3D numpy array.
        direction: Axis along which to extract RGB images ('z', 'y', 'x').
        rgb_size: Number of frames before and after for RGB generation.

    Returns:
        List of (index, RGB image) tuples.
    """
    d, h, w = img_stack.shape
    rgb_images = []

    if direction == 'z':
        for z in range(rgb_size, d - rgb_size):
            red_frame = img_stack[z - rgb_size, :, :]
            green_frame = img_stack[z, :, :]
            blue_frame = img_stack[z + rgb_size, :, :]
            rgb_image = np.stack([red_frame, green_frame, blue_frame], axis=-1)
            rgb_images.append((z, np.clip(rgb_image, 0, 255).astype(np.uint8)))
    elif direction == 'y':
        for y in range(rgb_size, h - rgb_size):
            red_frame = img_stack[:, y - rgb_size, :]
            green_frame = img_stack[:, y, :]
            blue_frame = img_stack[:, y + rgb_size, :]
            rgb_image = np.stack([red_frame, green_frame, blue_frame], axis=-1)
            rgb_images.append((y, np.clip(rgb_image, 0, 255).astype(np.uint8)))
    elif direction == 'x':
        for x in range(rgb_size, w - rgb_size):
            red_frame = img_stack[:, :, x - rgb_size]
            green_frame = img_stack[:, :, x]
            blue_frame = img_stack[:, :, x + rgb_size]
            rgb_image = np.stack([red_frame, green_frame, blue_frame], axis=-1)
            rgb_images.append((x, np.clip(rgb_image, 0, 255).astype(np.uint8)))

    return rgb_images

def run_inference(model, img_stack, rgb_size, output_csv, conf):
    """
    Perform inference on synthetic RGB images generated from 3D image stacks.

    Args:
        model: YOLO model instance.
        img_stack: 3D numpy array.
        rgb_size: Number of frames before and after for RGB generation.
        output_csv: Path to output CSV file.
    """
    directions = {'z': 2, 'y': 1, 'x': 0}  # Directional labels
    d, h, w = img_stack.shape

    with open(output_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['x_Foram_Pix', 'y_Foram_Pix', 'z_Foram_Pix', 
                             'width', 'height', 'direction', 'conf'])

        for direction, label in directions.items():
            rgb_images = generate_rgb_images(img_stack, direction, rgb_size)
            for index, rgb_image in rgb_images:
                results = model(rgb_image, conf=conf)

                for result in results:
                    boxes = result.boxes.xyxy
                    confidences = result.boxes.conf

                    for box, conf in zip(boxes, confidences):
                        x1, y1, x2, y2 = map(int, box)
                        width = x2 - x1
                        height = y2 - y1

                        if direction == 'z':
                            x_center = (x1 + x2) // 2
                            y_center = (y1 + y2) // 2
                            csv_writer.writerow([x_center, y_center, index, width, height, label, round(conf.item(), 3)])
                        elif direction == 'y':
                            x_center = (x1 + x2) // 2
                            z_center = (y1 + y2) // 2
                            csv_writer.writerow([x_center, index, z_center, width, height, label, round(conf.item(), 3)])
                        elif direction == 'x':
                            y_center = (x1 + x2) // 2
                            z_center = (y1 + y2) // 2
                            csv_writer.writerow([index, y_center, z_center, width, height, label, round(conf.item(), 3)])

                print(f'Processed {direction}-slice {index + 1}.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on synthetic RGB images from 3D .tif files using YOLO and save results in a CSV.')
    parser.add_argument('input_file', type=str, help='Path to the input .tif file.')
    parser.add_argument('output_csv', type=str, help='Path to the output CSV file.')
    parser.add_argument('model_path', type=str, help='Path to the YOLO model file (e.g., yolov11n.pt).')
    parser.add_argument('rgb_size', type=int, help='Number of frames before and after for RGB generation.')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold for detections. Default is 0.25.')
    args = parser.parse_args()

    # Load the specified YOLO model
    model = YOLO(args.model_path, task='detect')

    # Load the 3D image stack
    img_stack = tiff.imread(args.input_file)

    tracker = EmissionsTracker()
    tracker.start()

    # Run inference
    run_inference(model, img_stack, args.rgb_size, args.output_csv, args.conf)

    emissions = tracker.stop()
    print('-----------------------------------------------------')
    print(f'Total CPU energy consumption CodeCarbon (Process): {tracker._total_cpu_energy.kWh * 1000:.2f} Wh')
    print(f'Total RAM energy consumption CodeCarbon (Process): {tracker._total_ram_energy.kWh * 1000:.2f} Wh')
    print(f'Total GPU energy consumption CodeCarbon (Process): {tracker._total_gpu_energy.kWh * 1000:.2f} Wh')
    print(f'Total Energy consumption CodeCarbon (Process): {tracker._total_energy.kWh * 1000:.2f} Wh')
    print(f'Emissions by CodeCarbon (Process): {emissions * 1000:.2f} gCO2e')
