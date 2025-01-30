import os
import argparse
import numpy as np
import tifffile as tiff
import cv2

def create_rgb_images_from_tiff(tif_path, output_folder, rgb_size):
    # Read the TIFF file into a numpy array (assuming it's a multi-page TIFF)
    tif_data = tiff.imread(tif_path)

    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Ensure there are enough frames for RGB conversion (need at least 3)
    if tif_data.shape[0] < (2 * rgb_size + 1) or tif_data.shape[1] < (2 * rgb_size + 1) or tif_data.shape[2] < (2 * rgb_size + 1):
        print("Not enough frames in TIFF file for RGB conversion.")
        return
    
    # Iterate through each frame, creating RGB images from consecutive frames
    for z in range(rgb_size, tif_data.shape[0] - rgb_size):
        # Extract frames i-rgb_size, i, and i+rgb_size for R, G, and B channels
        red_frame = tif_data[z - rgb_size,:,:]  # Frame i-rgb_size (Red channel)
        green_frame = tif_data[z,:,:]    # Frame i (Green channel)
        blue_frame = tif_data[z + rgb_size,:,:] # Frame i+rgb_size (Blue channel)

        # Stack the frames to create an RGB image
        rgb_image = np.stack([red_frame, green_frame, blue_frame], axis=-1)

        # Convert to uint8 (0-255) if necessary (since TIFFs are usually float or int)
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

        # Save the RGB image as PNG
        output_filename = os.path.join(output_folder, f'image_z_{z}.png')
        cv2.imwrite(output_filename, rgb_image)
        
    print("Finished extraction in direction z !")
        
    # Iterate through each frame, creating RGB images from consecutive frames
    for y in range(rgb_size, tif_data.shape[1] - rgb_size):
        # Extract frames i-rgb_size, i, and i+rgb_size for R, G, and B channels
        red_frame = tif_data[:,y - rgb_size,:]  # Frame i-rgb_size (Red channel)
        green_frame = tif_data[:,y,:]    # Frame i (Green channel)
        blue_frame = tif_data[:,y + rgb_size,:] # Frame i+rgb_size (Blue channel)

        # Stack the frames to create an RGB image
        rgb_image = np.stack([red_frame, green_frame, blue_frame], axis=-1)

        # Convert to uint8 (0-255) if necessary (since TIFFs are usually float or int)
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

        # Save the RGB image as PNG
        output_filename = os.path.join(output_folder, f'image_y_{y}.png')
        cv2.imwrite(output_filename, rgb_image)
        
    print("Finished extraction in direction y !")

    # Iterate through each frame, creating RGB images from consecutive frames
    for x in range(rgb_size, tif_data.shape[2] - rgb_size):
        # Extract frames i-rgb_size, i, and i+rgb_size for R, G, and B channels
        red_frame = tif_data[:,:,x - rgb_size]  # Frame i-rgb_size (Red channel)
        green_frame = tif_data[:,:,x]    # Frame i (Green channel)
        blue_frame = tif_data[:,:,x + rgb_size] # Frame i+rgb_size (Blue channel)

        # Stack the frames to create an RGB image
        rgb_image = np.stack([red_frame, green_frame, blue_frame], axis=-1)

        # Convert to uint8 (0-255) if necessary (since TIFFs are usually float or int)
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

        # Save the RGB image as PNG
        output_filename = os.path.join(output_folder, f'image_x_{x}.png')
        cv2.imwrite(output_filename, rgb_image)

    print("Finished extraction in direction x !")
    print(f'RGB images saved to {output_folder}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert grayscale TIFF frames to synthetic RGB images and save them as PNG images in the specified output folder.')
    parser.add_argument('tif_path', type=str, help='Path to the input TIFF file.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder for RGB images.')
    parser.add_argument('rgb_size', type=int, help='Indicate which frames before and after will be stacked with the actual one.')

    args = parser.parse_args()
    
    create_rgb_images_from_tiff(args.tif_path, args.output_folder, args.rgb_size)
    print(f'RGB images extracted and saved to {args.output_folder}')
