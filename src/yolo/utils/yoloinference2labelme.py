import os
import argparse
import cv2
import json
import shutil  # Import shutil for copying files
from ultralytics import YOLO

# Inference function
def run_inference(model, input_folder, output_folder, conf_threshold):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to read image: {img_path}")
                continue  # Skip to the next file if reading fails
            height, width, _ = image.shape

            # Perform inference
            results = model(image, conf=conf_threshold)

            # Create the JSON structure for LabelMe
            json_data = {
                "version": "0.4.10",
                "flags": {},
                "shapes": [],
                "imagePath": filename,
                "imageData": None,
                "imageHeight": height,
                "imageWidth": width,
                "text": ""
            }

            # Process results
            for result in results:
                boxes = result.boxes.xyxy  # Get bounding boxes
                confidences = result.boxes.conf  # Get confidence scores
                class_labels = result.boxes.cls  # Get class labels if available

                for box, conf, label in zip(boxes, confidences, class_labels):
                    x1, y1, x2, y2 = map(float, box)  # Ensure coordinates are floats
                    shape = {
                        "label": f"{model.names[int(label)]}", 
                        "text": "",
                        "points": [[x1, y1], [x2, y2]],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {}
                    }
                    json_data["shapes"].append(shape)

            # Save the JSON file
            json_filename = os.path.splitext(filename)[0] + '.json'
            json_path = os.path.join(output_folder, json_filename)
            with open(json_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=2)

            # Copy the image to the output folder
            output_image_path = os.path.join(output_folder, filename)
            shutil.copy(img_path, output_image_path)

            print(f'Processed {filename} and saved annotations to {json_path} and copied image to {output_image_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on a folder of images using a YOLO model and generate LabelMe-compatible JSON annotations.')
    parser.add_argument('input_folder', type=str, help='Path to the folder containing input images.')
    parser.add_argument('output_folder', type=str, help='Path to the folder where output images will be saved.')
    parser.add_argument('model_path', type=str, help='Path to the YOLO model file (e.g., yolov11n.pt).')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold for detections. Default is 0.3.')

    args = parser.parse_args()

    # Load the specified YOLO model
    model = YOLO(args.model_path)

    # Run inference with the specified confidence threshold
    run_inference(model, args.input_folder, args.output_folder, args.conf)
