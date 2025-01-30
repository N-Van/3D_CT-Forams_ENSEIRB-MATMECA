from matplotlib import cm
import napari
import pandas as pd
import tifffile
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import random
import argparse


# The CSV file has to be in format:
# id,x_Foram_Pix,y_Foram_Pix,z_Foram_Pix


def napari_view_clusters(tif_path, clusters_result_path, barycenters, rectangle):

    # Create a parser
    print(f"Fichier .csv passé en paramètre : {clusters_result_path}")

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

        # Rectangle edges coordinates
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
    # Display the clusters in Napari
    #--------------------------------------------------------------------------------------------------------------------------


    # Load points coordinates and clusters
    csv_path = clusters_result_path
    data = pd.read_csv(csv_path)

    # Change axes x and z
    z = data['x_Foram_Pix'].values
    y = data['y_Foram_Pix'].values
    x = data['z_Foram_Pix'].values
    points = np.stack([x, y, z], axis=1)

    # Extract clusters information
    clusters = data['cluster'].values


    # Function to generate random colors excluding red
    def generate_random_color():
        color = (random.random(), random.random(), random.random())
        while (color[0] >= 0.6 and color[0] <= 0.4 and color[0] <= 0.4):
            color = (random.random(), random.random(), random.random())
        return color
    
    if barycenters:

        clusters = data['cluster'].unique()
        cluster_centroids_stack = []

        for cluster in clusters:
            if cluster != -1:
                # Select cluster's points
                cluster_points = points[data['cluster'] == cluster]
                
                # Barycenter calculation
                centroid = np.mean(cluster_points, axis=0)
                
                cluster_centroids_stack.append(centroid)


        cluster_centroids_stack = np.stack(cluster_centroids_stack)

        point_colors = []
        for cluster in np.unique(clusters):
            if cluster == -1:
                pass
            else:
                # Other clusters with random colors
                point_colors.append(generate_random_color())

        with napari.gui_qt():
            viewer.add_points(cluster_centroids_stack, size=5, face_color=point_colors, name='Foram by cluster')

    else:
        # Noisy samples will be represented by red points (cluster -1)
        red_color = (1.0, 0.0, 0.0)  # Red (RGB)

        # Create a colormap for other clusters
        colors = plt.cm.get_cmap('tab20', len(np.unique(clusters)) - 1)  # without -1 (noise)

        point_colors = []

        # Clusters' color dictionary
        cluster_colors_dict = {}

        # Fill the dictionary
        for cluster in np.unique(clusters):
            if cluster == -1:
                # Noisy points in red
                cluster_colors_dict[cluster] = red_color
            else:
                # Other clusters with random colors
                cluster_colors_dict[cluster] = generate_random_color()


        # Assign colors to the points depending on their cluster
        for cluster in clusters:
            point_colors.append(cluster_colors_dict[cluster])

        # Check the colors of the clusters (in the terminal)
        for cluster, color in cluster_colors_dict.items():
            print(f"Cluster {cluster} a la couleur {color}")

        with napari.gui_qt():
            viewer.add_points(points, size=5, face_color=point_colors, name='Foram by cluster')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Open Napari to view the annotated forams on the images.')
    parser.add_argument('tif_path', type=str, help='Path to the input tif file containing the images.')
    parser.add_argument('clusters_result_path', type=str, help='Path to the input file containing the clustered points.')
    parser.add_argument("--barycenters", default=False, type=bool, help="Only display clusters' barycenters")
    parser.add_argument('--rectangle', default=False, type=bool, help='To draw the rectangle for the training dataset that has been partially annotated.')

    args = parser.parse_args()

    napari_view_clusters(args.tif_path, args.clusters_result_path, args.barycenters, args.rectangle)