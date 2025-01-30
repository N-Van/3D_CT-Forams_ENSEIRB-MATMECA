# Dataset Utilities for 2.5D Image Processing

This folder contains various utility scripts to preprocess and manipulate 2.5D image data. 

## Folder Structure

```
src/utils/dataset/2.5D/
├── copy_rgb_files.py
├── gray_to_rgb_png.py
├── README.md
├── TIFF_CSV_SAM_to_LABELME_RGB.py
├── TIFF_CSV_SAM_to_PNG_RGB.py
└── TIFF_to_PNG_RGB.py
```

### File Descriptions

1. **`copy_rgb_files.py`**  
   This script replace images in a folder by images to a from another folder. Useful when you want to replace images with 2.5D images.

   **Usage:**
   ```bash
   python copy_rgb_files.py <training_folder> <complete_folder>
   ```

   - `training_folder`: Path to the folder with fewer images.
   - `complete_folder`: Path to the folder with more images (the source for the missing images).

2. **`gray_to_rgb_png.py`**  
   This script stacks grayscale images from a folder into RGB images using a specified frame offset. Each set of three consecutive frames is combined into a single RGB image.

   **Usage:**
   ```bash
   python gray_to_rgb_png.py <input_folder> <output_folder> <rgb_size>
   ```

   - `input_folder`: Folder containing the grayscale PNG files.
   - `output_folder`: Folder where the RGB images will be saved.
   - `rgb_size`: The number of frames above and below the current frame to create the RGB channels.

3. **`TIFF_CSV_SAM_to_LABELME_RGB.py`**  
   This script processes a TIFF file, and generates LabelMe annotations for the detected objects (e.g., foraminifera). The output includes RGB images and corresponding JSON annotations.

   **Usage:**
   ```bash
   python TIFF_CSV_SAM_to_LABELME_RGB.py <tif_path> <csv_path> <output_folder> [--box_width BOX_WIDTH] [--num_frames NUM_FRAMES] [--model_path MODEL_PATH] [--base_image_name BASE_NAME] [--apply_sam]
   ```

   - `tif_path`: Path to the TIFF file.
   - `csv_path`: Path to the CSV file containing coordinates of objects.
   - `output_folder`: Folder where the RGB images and annotations will be saved.
   - `--box_width`: Width of the bounding box around each point.
   - `--num_frames`: Number of frames above and below to consider.
   - `--model_path`: Path to the SAM model file (default: `mobile_sam.pt`).
   - `--base_image_name`: Base name for the output image and annotations.
   - `--apply_sam`: Flag to enable SAM-based segmentation for refining bounding boxes.

4. **`TIFF_CSV_SAM_to_PNG_RGB.py`**  
   Similar to the previous script, this tool processes TIFF images, and saves images with bounding boxes.

   **Usage:**
   ```bash
   python TIFF_CSV_SAM_to_PNG_RGB.py <tif_path> <csv_path> <output_folder> <box_width> <num_frames>
   ```

   - `tif_path`: Path to the TIFF file.
   - `csv_path`: Path to the CSV file with coordinates of objects.
   - `output_folder`: Folder to save the output PNG images with masks and bounding boxes.
   - `box_width`: Width of the bounding box for the objects.
   - `num_frames`: Number of frames above and below to consider.

5. **`TIFF_to_PNG_RGB.py`**  
   This script converts a multi-page TIFF file into a series of RGB PNG images by combining three consecutive frames. It supports creating RGB images in both the Z, Y and X  directions.

   **Usage:**
   ```bash
   python TIFF_to_PNG_RGB.py <tif_path> <output_folder> <rgb_size>
   ```

   - `tif_path`: Path to the TIFF file.
   - `output_folder`: Folder where the output RGB PNG images will be saved.
   - `rgb_size`: The number of frames above and below the current frame to create the RGB channels.

---

## Requirements

- Python 3.x
- Required libraries:
  - `numpy`
  - `opencv-python`
  - `pandas`
  - `torch`
  - `ultralytics` (for SAM model)
  - `tifffile`
  - `PIL`
  
You can install the required libraries using `pip`:
```bash
pip install numpy opencv-python pandas torch ultralytics tifffile Pillow
```

