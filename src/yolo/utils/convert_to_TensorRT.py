from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("/mnt/project/semestre/3D-CT-Forams_ENSEIRB-MATMECA/models/trained/yolov11l_2048_erode2.pt", task="detect")

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolov8n.engine'

