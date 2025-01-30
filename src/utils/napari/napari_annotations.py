from matplotlib import cm
import napari
import pandas as pd
import tifffile
import numpy as np
import argparse


# The CSV file has to be in format:
# id,x_Foram_Pix,y_Foram_Pix,z_Foram_Pix



def napari_annotation(tif_path, annotated_data_file_path, rectangle):

    # Napari viewer initialisation
    viewer = napari.Viewer()


    #--------------------------------------------------------------------------------------------------------------------------
    # Display the images in Napari
    #--------------------------------------------------------------------------------------------------------------------------

    # Load .tif images
    image_path = tif_path
    images = tifffile.imread(image_path)

    # Add images to the Napari viewer
    viewer.add_image(images, colormap='gray')

    

    #--------------------------------------------------------------------------------------------------------------------------
    # Display the anotated rectangle in Napari
    #--------------------------------------------------------------------------------------------------------------------------

    if rectangle == True:

        # Rectangle edges scoordinates
        rectangle = np.array([[362, 362], [1538, 362], [1538, 1538], [362, 1538]])

        # Add the polygon (rectangle)
        shapes_layer = viewer.add_shapes(
            rectangle,
            shape_type = 'polygon',
            edge_width = 5,
            edge_color = 'blue',
            face_color = 'None',
            blending   = 'minimum'
        )


    #--------------------------------------------------------------------------------------------------------------------------
    # Display the points corresponding to the annotations in Napari
    #--------------------------------------------------------------------------------------------------------------------------

    file_path = annotated_data_file_path
    data = pd.read_csv(file_path)

    z = data['x_Foram_Pix'].values
    y = data['y_Foram_Pix'].values
    x = data['z_Foram_Pix'].values 

    points = np.stack([x, y, z], axis=1)
    with napari.gui_qt():
        viewer.add_points(points, size=5, face_color='red', name='Foram Points')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Open Napari to view the annotated forams on the images.')
    parser.add_argument('tif_path', type=str, help='Path to the input tif file containing the images.')
    parser.add_argument('annotated_data_file_path', type=str, help='Path to the input file containing the locations of the annotated forams.')
    parser.add_argument('--rectangle', default=False, type=bool, help='To draw the rectangle for the training dataset that has been partially annotated.')

    args = parser.parse_args()

    napari_annotation(args.tif_path, args.annotated_data_file_path, args.rectangle)