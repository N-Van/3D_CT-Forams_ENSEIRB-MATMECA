import os
import argparse
import shutil

def replace_images(training_folder, complete_folder):
    """
    Replace images in the training folder with corresponding images from the complete folder.

    Args:
        training_folder (str): Path to the folder containing fewer images.
        complete_folder (str): Path to the folder containing more images.
    """
    # Get the list of image files in the training folder
    training_images = {f for f in os.listdir(training_folder) if os.path.isfile(os.path.join(training_folder, f))}

    for training_image in training_images:
        complete_image_path = os.path.join(complete_folder, training_image)
        training_image_path = os.path.join(training_folder, training_image)

        if os.path.exists(complete_image_path):
            # Copy the corresponding image from the complete folder to the training folder
            shutil.copy(complete_image_path, training_image_path)
            print(f"Replaced {training_image} with {complete_image_path}.")
        else:
            print(f"Image {training_image} not found in {complete_folder}, skipping.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace images in the training folder with corresponding images from the complete folder.")
    parser.add_argument("training_folder", type=str, help="Path to the folder containing fewer images.")
    parser.add_argument("complete_folder", type=str, help="Path to the folder containing more images.")

    args = parser.parse_args()

    replace_images(args.training_folder, args.complete_folder)
