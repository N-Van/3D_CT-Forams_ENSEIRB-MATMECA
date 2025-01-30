import os
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

# Inference function
def run_inference(model, input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)

            # Perform inference
            results = model(image, conf=0.3)

            # Process results
            for result in results:
                boxes = result.boxes.xyxy  # Get bounding boxes
                confidences = result.boxes.conf  # Get confidence scores
                class_ids = result.boxes.cls  # Get class IDs

                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = model.names[int(class_id)]  # Get class name from ID
                    label = f'{class_name} {conf:.2f}'

                    # Draw bounding box and label
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save the output image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image)
            print(f'Processed {filename} and saved to {output_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on a folder of images using YOLO.')
    parser.add_argument('input_folder', type=str, help='Path to the folder containing input images.')
    parser.add_argument('output_folder', type=str, help='Path to the folder where output images will be saved.')
    parser.add_argument('model_path', type=str, help='Path to the YOLO model file (e.g., yolov11n.pt).')

    args = parser.parse_args()

    # Load the specified YOLO model
    model = YOLO(args.model_path)

    # Run inference
    run_inference(model, args.input_folder, args.output_folder)
