import os
from PIL import Image
import argparse

def create_gif(input_folder, output_path, duration=100, resize_factor=0.5, reduce_colors=False):
    """
    Create a GIF from images in a folder with reduced quality.

    Args:
        input_folder (str): Path to the folder containing images.
        output_path (str): Path where the GIF will be saved.
        duration (int): Duration between frames in milliseconds. Default is 100ms.
        resize_factor (float): Factor by which to resize the images (e.g., 0.5 for half size).
        reduce_colors (bool): Whether to reduce the color depth to 256 colors.
    """
    # Get list of image files sorted numerically
    image_files = sorted(
        [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])
    )
    
    if not image_files:
        print("No images found in the folder!")
        return

    # Open images and reduce quality
    images = []
    for img_path in image_files:
        img = Image.open(img_path)
        
        # Resize the image
        if resize_factor < 1.0:
            new_size = (int(img.width * resize_factor), int(img.height * resize_factor))
            img = img.resize(new_size)
        
        # Reduce color depth
        if reduce_colors:
            img = img.convert("P", palette=Image.ADAPTIVE, colors=256)

        images.append(img)

    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=1
    )
    print(f"GIF saved at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create a GIF from images in a folder with reduced quality.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing images.")
    parser.add_argument("output_path", type=str, help="Path where the GIF will be saved.")
    parser.add_argument("--duration", type=int, default=100, help="Duration between frames in milliseconds (default: 100ms).")
    parser.add_argument("--resize_factor", type=float, default=0.5, help="Factor to resize the images (default: 0.5).")
    parser.add_argument("--reduce_colors", action="store_true", help="Reduce color depth to 256 colors.")

    args = parser.parse_args()

    create_gif(
        input_folder=args.input_folder,
        output_path=args.output_path,
        duration=args.duration,
        resize_factor=args.resize_factor,
        reduce_colors=args.reduce_colors
    )

if __name__ == "__main__":
    main()
