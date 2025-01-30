import os
import argparse
import csv
import numpy as np
import tifffile as tiff
from ultralytics import YOLO
from codecarbon import EmissionsTracker

# Inference function
def run_inference(model, input_file, output_csv, conf_threshold):
    # Load the 3D image
    img_stack = tiff.imread(input_file)
    d, h, w = img_stack.shape  # Dimensions of the 3D image

    # Prepare CSV for output
    with open(output_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['x_Foram_Pix', 'y_Foram_Pix', 'z_Foram_Pix', 
                             'width', 'height', 'direction', 'conf'])

        # Process slices along the Z-axis
        for z in range(d):
            image = img_stack[z, :, :]  # 2D slice in the Z direction

            # Convert grayscale image to 3-channel RGB format
            image_rgb = np.stack((image,) * 3, axis=-1)

            # Perform inference
            results = model(image_rgb, conf=conf_threshold)

            for result in results:
                boxes = result.boxes.xyxy
                confidences = result.boxes.conf

                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    x_center = (x1 + x2) // 2
                    y_center = (y1 + y2) // 2
                    width = x2 - x1
                    height = y2 - y1
                    csv_writer.writerow([x_center, y_center, z, width, height, 2, round(conf.item(), 3)])

            print(f'Processed slice {z+1}/{d} in Z direction.')

        # Process slices along the X-axis
        for x in range(h):
            image = img_stack[:, :, x]  # 2D slice in the X direction

            # Convert grayscale image to 3-channel RGB format
            image_rgb = np.stack((image,) * 3, axis=-1)

            results = model(image_rgb, conf=conf_threshold)

            for result in results:
                boxes = result.boxes.xyxy
                confidences = result.boxes.conf

                for box, conf in zip(boxes, confidences):
                    y1, z1, y2, z2 = map(int, box)
                    z_center = (z1 + z2) // 2
                    y_center = (y1 + y2) // 2
                    width = y2 - y1
                    height = z2 - z1
                    csv_writer.writerow([x, y_center, z_center, width, height, 0, round(conf.item(), 3)])

            print(f'Processed slice {x+1}/{h} in X direction.')

        # Process slices along the Y-axis
        for y in range(w):
            image = img_stack[:, y, :]  # 2D slice in the Y direction

            # Convert grayscale image to 3-channel RGB format
            image_rgb = np.stack((image,) * 3, axis=-1)

            results = model(image_rgb, conf=conf_threshold)

            for result in results:
                boxes = result.boxes.xyxy
                confidences = result.boxes.conf

                for box, conf in zip(boxes, confidences):
                    x1, z1, x2, z2 = map(int, box)
                    z_center = (z1 + z2) // 2
                    x_center = (x1 + x2) // 2
                    width = x2 - x1
                    height = z2 - z1
                    csv_writer.writerow([x_center, y, z_center, width, height, 1, round(conf.item(), 3)])

            print(f'Processed slice {y+1}/{w} in Y direction.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on a 3D .tif file using YOLO and save results in a CSV.')
    parser.add_argument('input_file', type=str, help='Path to the input .tif file.')
    parser.add_argument('output_csv', type=str, help='Path to the output CSV file.')
    parser.add_argument('model_path', type=str, help='Path to the YOLO model file (e.g., yolov11n.pt).')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold for detections. Default is 0.25.')

    args = parser.parse_args()

    # Load the specified YOLO model
    model = YOLO(args.model_path, task='detect') 
    
    tracker = EmissionsTracker()
    tracker.start()

    # Run inference with the specified confidence threshold
    run_inference(model, args.input_file, args.output_csv, args.conf)

    emissions: float = tracker.stop()
    print('-----------------------------------------------------')
    print('Total CPU energy consumption CodeCarbon (Process): ' + str(tracker._total_cpu_energy.kWh*1000) + ' Wh')
    print('Total RAM energy consumption CodeCarbon (Process): ' + str(tracker._total_ram_energy.kWh*1000) + ' Wh')
    print('Total GPU energy consumption CodeCarbon (Process): ' + str(tracker._total_gpu_energy.kWh*1000) + ' Wh')
    print('Total Energy consumption CodeCarbon (Process): ' + str(tracker._total_energy.kWh*1000) + ' Wh')
    print('Emissions by CodeCarbon (Process): '+ str(emissions*1000) + ' gCO2e')
