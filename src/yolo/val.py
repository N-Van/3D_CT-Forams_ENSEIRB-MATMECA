import argparse
import csv
from ultralytics import YOLO

def main(model_path, data_path, output_csv):
    # Load the model
    model = YOLO(model_path)  # load the specified model

    # Validate the model
    metrics = model.val(data=data_path, conf=0.6)  # specify dataset path

    # Extract all metrics
    results = metrics.results_dict  # Directly get all metrics as a dictionary

    # Save results to CSV
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write header
        header = ["Metric", "Value"]
        writer.writerow(header)

        # Write metrics
        for metric, value in results.items():
            writer.writerow([metric, value])

    print(f"Validation results saved to {output_csv}")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Validate a YOLO model and save results to CSV.")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO model (e.g., best.pt).")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset YAML file.")
    parser.add_argument("--output", type=str, default="validation_results.csv", help="Path to save the output CSV file.")

    args = parser.parse_args()

    # Run the main function
    main(args.model, args.data, args.output)
