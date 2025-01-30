import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd
import argparse
import os


def calculate_barycenters(prediction_clusters):
    # Lecture du file CSV
    df = pd.read_csv(prediction_clusters)

    if len(df['cluster']) != 0:


        # Calcul des barycentres pour chaque cluster
        barycenters = (
            df.groupby('cluster')
            .apply(lambda group: pd.Series({
                'x_Barycenter': int(round(group['x_Foram_Pix'].mean())),
                'y_Barycenter': int(round(group['y_Foram_Pix'].mean())),
                'z_Barycenter': int(round(group['z_Foram_Pix'].mean()))
            }))
            .reset_index()
        )

        # Sauvegarde des résultats dans un nouveau file CSV
        output_csv = "results/clusters_barycenters.csv"
        barycenters.to_csv(output_csv, index=False)





def merge(prediction_clusters, annotations_file, threshold):

    # Extract the filename without the extension
    filename = os.path.splitext(os.path.basename(prediction_clusters))[0]
    # Remove the prefix "hdbscan_"
    filename = filename.replace("hdbscan_", "")
    # Split the string by '_'
    parts = filename.split('_') # ['HITL', '2D', '5', '6']
    if threshold >= 0 :
        dataset = parts[0]
        approach = parts[1]
        minClusterSize = parts[2]
        minSamples = parts[3]
    else:
        dataset = parts[1]
        approach = parts[2]
        minClusterSize = parts[3]
        minSamples = parts[4]


    # Load CSV files
    file1 = pd.read_csv(annotations_file)
    file2 = pd.read_csv('results/clusters_barycenters.csv')
    filtered_file2 = file2[file2['cluster'] != -1]

    # Extract coordinates and id
    points1 = file1[['x_Foram_Pix', 'y_Foram_Pix', 'z_Foram_Pix']].values
    points2 = filtered_file2[['x_Barycenter', 'y_Barycenter', 'z_Barycenter']].values

    ids1 = file1['id'].values
    clusters2 = file2['cluster'].values


    # Calculate the distance matrix
    distances = np.zeros((len(points1), len(points2)))
    for i, point1 in enumerate(points1):
        for j, point2 in enumerate(points2):
            distances[i, j] = np.linalg.norm(point1 - point2)

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(distances)

    results = []
    for i, j in zip(row_ind, col_ind):
        point1_id = ids1[i]
        point2_cluster = clusters2[j]
        point1_coords = tuple(map(int, points1[i]))
        point2_coords = tuple(map(int, points2[j]))
        distance = int(round(distances[i, j]))
        if distance < 30:
            results.append({
                "id (file1)": point1_id,
                "cluster (file2)": point2_cluster,
                "Point file1 (x, y, z)": point1_coords,
                "Point file2 (x, y, z)": point2_coords,
                "Distance": distance
            })

    # Convert into a DataFrame
    df_results = pd.DataFrame(results)

    # Save as CSV file
    if threshold == 0:
        output_file = f'results/associations_points_{dataset}_{approach}_{minClusterSize}_{minSamples}.csv'
    else:
        output_file = f'results/associations_points_{dataset}_{approach}_{minClusterSize}_{minSamples}_{threshold}.csv'
    df_results.to_csv(output_file, index=False)

    print(f"The results have been saved to the file : {output_file}")


    

    truePositives = len(results)
    falsePositives = max(len(filtered_file2) - truePositives, 0)
    data = {
        "Dataset": [dataset],
        "Model": [approach],
        "minClusterSize": [minClusterSize],
        "minSamples": [minSamples],
        "threshold": [threshold],
        "Nb vrais positifs": [truePositives],
        "Nb faux positifs": [falsePositives],
    }
    df = pd.DataFrame(data)
    output_file = "results/results_to_compare.csv"
    if os.path.exists(output_file):
        df.to_csv(output_file, mode="a", index=False, header=False)
    else:
        df.to_csv(output_file, index=False, header=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the barycenters points for each cluster and save into clusters_barycenters.csv.')
    parser.add_argument('prediction_clusters', type=str, help='Path to the input csv file containing the predictions.')
    parser.add_argument('annotations_file', type=str, help='Path to the input csv file containing the annotations.') # '../../datasets/RAW/1/Aq3T1_filtered_data.csv'
    parser.add_argument('--threshold', default=0, type=float, help='Confidence threshold.')

    args = parser.parse_args()

    calculate_barycenters(args.prediction_clusters)
    merge(args.prediction_clusters, args.annotations_file, args.threshold)