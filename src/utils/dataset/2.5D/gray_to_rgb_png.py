import os
import argparse
import numpy as np
import cv2

def create_rgb_images_from_pngs(input_folder, output_folder, rgb_size):
    # Liste triée des fichiers PNG dans le dossier d'entrée
    filenames = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    png_files = sorted(filenames, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    

    # Vérifiez que le nombre d'images est suffisant
    if len(png_files) < (2 * rgb_size + 1):
        print("Pas assez d'images dans le dossier pour effectuer l'empilement RGB.")
        return

    # Vérifiez que le dossier de sortie existe, sinon, créez-le
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Parcourez les fichiers en respectant la taille du décalage rgb_size
    for z in range(rgb_size, len(png_files) - rgb_size):
        # Chargez les images pour les canaux R, G et B
        red_frame = cv2.imread(os.path.join(input_folder, png_files[z - rgb_size]), cv2.IMREAD_GRAYSCALE)
        green_frame = cv2.imread(os.path.join(input_folder, png_files[z]), cv2.IMREAD_GRAYSCALE)
        blue_frame = cv2.imread(os.path.join(input_folder, png_files[z + rgb_size]), cv2.IMREAD_GRAYSCALE)

        # Empilez les images pour créer une image RGB
        rgb_image = np.stack([red_frame, green_frame, blue_frame], axis=-1)

        # Sauvegardez l'image RGB dans le dossier de sortie
        output_filename = os.path.join(output_folder, f'image_z_{z}.png')
        cv2.imwrite(output_filename, rgb_image)

    print(f"Images RGB empilées enregistrées dans {output_folder}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Empile des images PNG en entrée pour créer des images RGB selon l\'axe Z.')
    parser.add_argument('input_folder', type=str, help='Dossier contenant les images PNG en entrée.')
    parser.add_argument('output_folder', type=str, help='Dossier où les images RGB seront enregistrées.')
    parser.add_argument('rgb_size', type=int, help='Taille du décalage pour les canaux R et B.')

    args = parser.parse_args()

    create_rgb_images_from_pngs(args.input_folder, args.output_folder, args.rgb_size)
