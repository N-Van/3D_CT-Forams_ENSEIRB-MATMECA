import os
import subprocess

# Chemins des fichiers d'entrée
INPUT_HITL = "../../../RESULTS/2D/HITL_2D.csv"
INPUT_RAW = "../../datasets/RAW/1/Aq3T1_filtered_data.csv"

a = 15
b = 15
values_to_test = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]

# Boucle sur les valeurs de threshold
for threshold in values_to_test:
        print(f"Execution avec minClusterSize={a}, minSamples={b} et threshold={threshold}")

        # Commande pour hdbscan_clustering.py
        hdbscan_command = [
            "python", "hdbscan_clustering_with_confidence_before.py", INPUT_HITL, INPUT_RAW,
            "--minClusterSize", str(a), "--minSamples", str(b), "--threshold",  str(threshold)
        ]
        subprocess.run(hdbscan_command)

        # Nom du fichier de résultats généré par hdbscan_clustering.py
        if threshold != 0:
             OUTPUT_HDBSCAN = f"results/hdbscan_HITL_2D_{a}_{b}_{threshold}.csv"
        else:
             OUTPUT_HDBSCAN = f"results/hdbscan_HITL_2D_{a}_{b}.csv"

        # Vérifie si le fichier de sortie existe avant de lancer merger.py
        if os.path.isfile(OUTPUT_HDBSCAN):
            merger_command = ["python", "merger.py", OUTPUT_HDBSCAN, INPUT_RAW, "--threshold", str(threshold)]
            subprocess.run(merger_command)
        else:
            print(f"Fichier de sortie {OUTPUT_HDBSCAN} non trouvé, saut de la fusion.")
