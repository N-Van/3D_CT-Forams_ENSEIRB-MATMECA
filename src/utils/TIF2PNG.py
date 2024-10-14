import os
import argparse
from PIL import Image

def extract_images(tif_path, output_folder):
    # Open the TIFF file
    with Image.open(tif_path) as img:
        # Check if the output folder exists, if not, create it
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Iterate through the frames of the TIFF file
        for i in range(img.n_frames):
            img.seek(i)  # Move to the ith frame
            img.save(os.path.join(output_folder, f'image_{i + 1}.png'))  # Save as PNG

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract individual frames from a multi-frame TIFF file and save them as PNG images in the specified output folder.')
    parser.add_argument('tif_path', type=str, help='Path to the input TIFF file.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder.')

    args = parser.parse_args()
    
    extract_images(args.tif_path, args.output_folder)
    print(f'Images extracted to {args.output_folder}')
