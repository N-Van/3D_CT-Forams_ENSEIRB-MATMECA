# HDBSCAN Clustering with Confidence Filtering

## Description
This directory contains multiple scripts for clustering foram detection data using HDBSCAN and analyzing results by comparing different confidence filtering strategies.

## Directory Structure
```
├── clustering
│   ├── hdbscan_clustering.py
│   ├── hdbscan_clustering_with_confidence_after.py
│   ├── hdbscan_clustering_with_confidence_before.py
│   ├── merger.py
│   ├── pipeline_compare_confidence.py
│   └── README.md
```

## Scripts

### `hdbscan_clustering.py`
This script applies the HDBSCAN algorithm to a predictions file and generates a CSV file containing detected clusters.
- Inputs:
  - `prediction_file_path`: Path to the CSV file containing predictions.
  - `annotated_data_file_path`: Path to the CSV file containing annotations.
  - `--minClusterSize`: Minimum number of points required to form a cluster.
  - `--minSamples`: Minimum number of points required for a core point.
- Outputs:
  - CSV file with detected clusters.
  - `trace.txt` file summarizing the results.

### `hdbscan_clustering_with_confidence_before.py` & `hdbscan_clustering_with_confidence_after.py`
These scripts apply HDBSCAN with confidence-based filtering, but in a different order.

| Feature | Before Filtering (`hdbscan_clustering_with_confidence_before.py`) | After Filtering (`hdbscan_clustering_with_confidence_after.py`) |
|---------|-------------------------------------------------|------------------------------------------------|
| Confidence Filtering | Applied **before** clustering | Applied **after** clustering |
| Noise Reduction | May discard valid points before clustering | More refined filtering based on cluster confidence |
| Output | Clusters are formed only from high-confidence points | All points are clustered first, then unreliable clusters are removed |

- Inputs are similar to `hdbscan_clustering.py`, with an additional `--threshold` parameter.
- Outputs:
  - `results/hdbscan_<filename>_<minClusterSize>_<minSamples>.csv`: Raw clustering results.
  - `results/hdbscan_<filename>_<minClusterSize>_<minSamples>_<threshold>.csv`: Clustering results with confidence filtering.
  - `results/filtered_hdbscan_<filename>_<minClusterSize>_<minSamples>_<threshold>.csv`: Only for the **after filtering** version, containing high-confidence clusters.
  - `results/trace.txt`: Log file recording clustering results and parameters.

### `pipeline_compare_confidence.py`
This script runs `hdbscan_clustering_with_confidence_before.py` for different threshold values and merges results if an output file is generated.
- Inputs:
  - `INPUT_HITL`: Predictions CSV file.
  - `INPUT_RAW`: Annotations CSV file.
  - List of threshold values to test.
- Outputs:
  - Automatic execution of `hdbscan_clustering_with_confidence_before.py`.
  - Verification and merging of results using `merger.py`.

### `merger.py`
This script merges clustering results with annotations to compare clustering performance.
- Inputs:
  - `prediction_clusters`: CSV file containing clustering results.
  - `annotations_file`: CSV file with ground truth annotations.
  - `--threshold`: Confidence threshold (optional).
- Features:
  - Computes cluster centroids.
  - Compares results with annotations using minimum distance.
  - Saves results in `results_to_compare.csv`.

## Usage
Run the scripts in the following order for a complete analysis:

1. Apply clustering:
   ```bash
   python src/clustering/hdbscan_clustering_with_confidence_before.py <predictions.csv> <annotations.csv> --minClusterSize 5 --minSamples 5 --threshold 0.7
   ```
   ```bash
   python src/clustering/hdbscan_clustering_with_confidence_after.py <predictions.csv> <annotations.csv> --minClusterSize 5 --minSamples 5 --threshold 0.7
   ```
2. Compare different confidence threshold values:
   ```bash
   python pipeline_compare_confidence.py
   ```
3. Merge and analyze results:
   ```bash
   python merger.py <clustering_file> <annotations_file>
   ```

## Interpretation
- Noise points are labeled `-1`.
- The final number of detected foram clusters is compared with ground truth annotations.
- The **after filtering** method may better retain valid foram clusters while removing uncertain detections.

## Notes
- Use `hdbscan_clustering_with_confidence_after.py` if you want to keep all detections and filter unreliable clusters post hoc.
- Prefer `hdbscan_clustering_with_confidence_before.py` if you only want to cluster high-confidence points from the start.
