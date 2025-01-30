import pandas as pd
import numpy as np
import hdbscan
import argparse
from os import makedirs, path
import os


def apply_hdbscan(prediction_file_path, annotated_data_file_path, minSamples, minClusterSize, threshold):

    if not path.exists("./results"):
        makedirs("./results")
    if not path.exists("./results/trace.txt"):
        result_file = open("results/trace.txt", "x")
        result_file.close()
    

    print(f"HYPERPARAMETERS:\n-minClusterSize = {minClusterSize}\n-minSamples = {minSamples}")


    # Load the points from the .csv file into a DataFrame
    points_df = pd.read_csv(prediction_file_path)
    points = points_df[['x_Foram_Pix', 'y_Foram_Pix', 'z_Foram_Pix']].values
    
    # Filter the points where the 'conf' column is greater than or equal to the threshold
    filtered_before_df = points_df[points_df['conf'] >= threshold]

    # Extract the 'x_Foram_Pix', 'y_Foram_Pix', 'z_Foram_Pix' coordinates
    points = filtered_before_df[['x_Foram_Pix', 'y_Foram_Pix', 'z_Foram_Pix']].values

    

    # Instanciate HDBSCAN model
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size = minClusterSize,     # The minimum size of a cluster.
        min_samples = minSamples,              # Minimal number of neighbours for a point to be considered as a cluster center.
        alpha = 0.8,                           # The alpha value to set the weight of the mutual reachability distance between two points.
        metric = 'euclidean',                  # Distance used for the clustering
        cluster_selection_method = 'leaf'       # Method used to defin the clusters

    )

    # Apply HDBSCAN
    cluster_labels = clusterer.fit_predict(points)


    # Add clusters' labels to a copy of the points dataFrame
    df_with_clusters = filtered_before_df.copy()
    df_with_clusters['cluster'] = cluster_labels


    # Save results into a csv file
    filename = os.path.splitext(os.path.basename(prediction_file_path))[0]
    df_with_clusters.to_csv(f"results/hdbscan_{filename}_{minClusterSize}_{minSamples}.csv", index=False)

    if threshold == 0:
        df_with_clusters.to_csv(f"results/hdbscan_{filename}_{minClusterSize}_{minSamples}.csv", index=False)
    else:
        df_with_clusters.to_csv(f"results/hdbscan_{filename}_{minClusterSize}_{minSamples}_{threshold}.csv", index=False)


    # # Charger le fichier results.csv
    # df_clustered = pd.read_csv(f"results/hdbscan_{filename}_{minClusterSize}_{minSamples}.csv")

    # # Calculer la moyenne de `conf` pour chaque cluster
    # mean_conf_per_cluster = df_clustered.groupby('cluster')['conf'].mean()

    # # Trouver les clusters dont la moyenne des conf est inférieure au seuil
    # clusters_to_remove = mean_conf_per_cluster[mean_conf_per_cluster < threshold].index

    # # Filtrer les lignes du dataframe pour ne garder que celles qui ne sont pas dans les clusters à retirer
    # filtered_df = df_clustered[~df_clustered['cluster'].isin(clusters_to_remove)]

    # # Sauvegarder le résultat dans un nouveau fichier CSV
    # filtered_df.to_csv('results/filtered_results.csv', index=False)

    # if threshold != 0:
    #     filtered_df.to_csv(f"results/filtered_hdbscan_{filename}_{minClusterSize}_{minSamples}_{threshold}.csv", index=False)



    # Calculate the number of clusters
    num_clusters = 0
    data_with_clusters = df_with_clusters[['x_Foram_Pix','y_Foram_Pix','z_Foram_Pix', 'cluster']].values
    col = [ligne[3] for ligne in data_with_clusters]
    num_clusters = len(np.unique(col)) - (1 if -1 in col else 0)

    # Print
    print(f"Number of noise points (label -1) : {np.sum(cluster_labels == -1)}")
    print(f"Number of detected foram : {num_clusters}")


    # Number of foram counted/annotated in the dataset
    df_annotated = pd.read_csv(annotated_data_file_path)
    annotated_data = df_annotated[['x_Foram_Pix','y_Foram_Pix','z_Foram_Pix']].values
    num_annotated_elements = 0
    for pt in annotated_data:
        num_annotated_elements += 1
    print(f"Number of annotated foram: {num_annotated_elements}")

    # Calculate the difference between the number of clusters found and the number of foram annotated
    difference = abs(num_clusters - num_annotated_elements)
    print(f"Difference between the number of clusters and the number of annotated foram: {difference}")


    result_file = open("results/trace.txt", "a")
    result_file.write(f"-------------HDBSCAN-------------\n")
    result_file.write(f"Predictions: {prediction_file_path}\nAnnotations: {annotated_data_file_path}\n")
    result_file.write(f"minClusterSize = {minClusterSize}\nminSamples = {minSamples}\n")
    result_file.write(f"threshold = {threshold}\n")
    result_file.write(f"Number of noise points (label -1) : {np.sum(cluster_labels == -1)}\n")
    result_file.write(f"Number of detected clusters : {num_clusters}\n")
    result_file.write(f"Number of annotated foram: {num_annotated_elements}\n")
    result_file.write(f"Difference between the number of clusters and the number of annotated foram: {difference}\n")
    result_file.write("---------------------------------")
    result_file.write(f"\n\n")
    result_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply HDBSCAN on the file of predicted points.')
    parser.add_argument('prediction_file_path', type=str, help='Path to the input csv file containing the predictions.')
    parser.add_argument('annotated_data_file_path', type=str, help='Path to the input file containing the locations of the annotated forams.')
    parser.add_argument('--minClusterSize', default=5, type=int, help='Minimum number of points required to form a cluster defined for HDBSCAN.')
    parser.add_argument('--minSamples', default=5, type=int, help='Minimum number of points required to consider a point as a core point defined for HDBSCAN.')
    parser.add_argument('--threshold', default=0.7, type=float, help='Confidence threshold.')
    
    args = parser.parse_args()

    apply_hdbscan(args.prediction_file_path, args.annotated_data_file_path, args.minSamples, args.minClusterSize, args.threshold)