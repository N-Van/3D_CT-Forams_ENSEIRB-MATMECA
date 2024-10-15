# Datasets Directory

This directory contains various datasets used for training and testing the foraminifera detection model.

## Subdirectories

- **LABELME**: Contains labeled data in LabelMe format for training and evaluation.  

- **RAW**: Contains the raw microtomography images in TIF or PNG format.

- **YOLO**: Contains files specific to YOLO dataset preparation, including an example dataset configuration.  

## Dataset Annotations Format 

-  **LabelMe Format**: LabelMe annotations are stored in JSON files, with each file corresponding to an image. The JSON file contains details about the shapes of the labeled objects, their coordinates, and other metadata.

  **Example of LabelMe JSON (Polygon):**
  ```json
  {
      "version": "4.5.7",
      "flags": {},
      "shapes": [
          {
              "label": "foraminifera",
              "points": [[120, 130], [150, 130], [150, 160], [120, 160]],
              "group_id": null,
              "shape_type": "polygon",
              "flags": {}
          }
      ],
      "imagePath": "image_01.jpg",
      "imageHeight": 400,
      "imageWidth": 400
  }
  ```

  **Example of LabelMe JSON (Rectangle):**
  ```json
  {
      "version": "4.5.7",
      "flags": {},
      "shapes": [
          {
              "label": "foraminifera",
              "points": [[100, 100], [200, 200]],
              "group_id": null,
              "shape_type": "rectangle",
              "flags": {}
          }
      ],
      "imagePath": "image_02.jpg",
      "imageHeight": 400,
      "imageWidth": 400
  }
  ```

  In this rectangle example, the points define the top-left and bottom-right corners of the rectangle where the object is located.

- **YOLO Format**: YOLO uses text files for annotations, where each line corresponds to an object in the image. The format includes the class ID and the normalized coordinates of the bounding box.

  **Example of YOLO Annotation:**
  ```
  0 0.375 0.375 0.25 0.25
  1 0.5 0.5 0.2 0.2
  ```
  Here, the first value is the class ID (0 for the first object and 1 for the second), followed by:
  - Normalized Center X
  - Normalized Center Y
  - Normalized Width
  - Normalized Height

  In this example:
  - The first object is of class ID 0, centered at (0.375, 0.375) with a width and height of 0.25.
  - The second object is of class ID 1, centered at (0.5, 0.5) with a width and height of 0.2.
