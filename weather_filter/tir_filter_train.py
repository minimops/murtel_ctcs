import os
import csv
import argparse
import joblib
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

"""
Script that:
 - calculates entropies of TIR images
 - selects best decision boundary

Usage example:
    python tir_filter_train.py \
        data/TIR_images \
        data/filtering/tir_labels.csv \
        --outputs_csv data/filtering/tir_filtering.csv \
        --threshold_file data/filtering/tir_entropy_threshold.joblib
"""

def compute_entropy(image_array):
    """
    Computes the Shannon entropy of a grayscale image array.
    """
    pixels = image_array.flatten()
    histogram, _ = np.histogram(pixels, bins=256, range=(0, 255))
    total_pixels = np.sum(histogram)
    if total_pixels == 0:
        return 0
    probs = histogram / total_pixels
    probs = probs[probs > 0]
    entropy_val = -np.sum(probs * np.log2(probs))
    return entropy_val


def load_images_and_labels(images_dir, csv_path):
    """
    Loads image filenames and labels from a CSV.
    """
    image_paths = []
    labels = []
    with open(csv_path, newline='') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)  # skip header if present
        for row in reader:
            if len(row) < 3:
                continue
            fname, tag = row[0], row[2]
            label = 0 if tag.lower() == 'x' else 1
            img_path = os.path.join(images_dir, fname)
            if not os.path.isfile(img_path):
                print(f"Warning: File '{img_path}' not found. Skipping.")
                continue
            image_paths.append(img_path)
            labels.append(label)
    return image_paths, labels


def find_best_threshold(entropies, labels, num_thresholds=100):
    """
    Finds the best threshold based on Youden's index (sensitivity + specificity - 1)
    returns (best_threshold, best_sensitivity, best_specificity)
    """
    if not entropies:
        return None, 0, 0
    e_min = min(entropies)
    e_max = max(entropies)
    thresholds = np.linspace(e_min, e_max, num_thresholds)

    best_threshold = None
    best_youden = -1
    best_sensitivity = 0
    best_specificity = 0

    for t in thresholds:
        predictions = [1 if e > t else 0 for e in entropies]
        TP = sum((p == 1 and l == 1) for p, l in zip(predictions, labels))
        FP = sum((p == 1 and l == 0) for p, l in zip(predictions, labels))
        FN = sum((p == 0 and l == 1) for p, l in zip(predictions, labels))
        TN = sum((p == 0 and l == 0) for p, l in zip(predictions, labels))

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        youden_index = sensitivity + specificity - 1

        if youden_index > best_youden:
            best_threshold = t
            best_youden = youden_index
            best_sensitivity = sensitivity
            best_specificity = specificity

    return best_threshold, best_sensitivity, best_specificity


def plot_entropy_distribution(entropies, labels, threshold):
    """
    Plots the distribution of entropies for "good" vs "bad" images,
    along with a vertical line showing the chosen threshold.
    """
    entropies_good = [e for e, l in zip(entropies, labels) if l == 1]
    entropies_bad = [e for e, l in zip(entropies, labels) if l == 0]

    bin_width = 0.2
    bins = np.arange(0, 8 + bin_width, bin_width)

    plt.hist(entropies_bad, bins=bins, alpha=0.5, label="Bad images (0)", color='orange')
    plt.hist(entropies_good, bins=bins, alpha=0.5, label="Good images (1)", color='green')
    plt.axvline(threshold, color='blue', linestyle=':', label=f'Threshold={threshold:.3f}')

    plt.title("Entropy Distributions")
    plt.xlabel("Entropy")
    plt.ylabel("Count")
    plt.legend()
    plt.show()


def main(images_dir,
        csv_path,
        output_csv="tir_labeled_predictions.csv",
        threshold_file="tir_entropy_threshold.joblib"):
    """
    Trains a simple threshold-based classifier for TIR images:
      - Loads labeled data
      - Computes entropies
      - Finds the best threshold (Youden's index)
      - Saves threshold to disk (joblib)
      - Writes a CSV with [image, label, entropy, predicted_label]

    :param images_dir: Directory of TIR images.
    :param csv_path: CSV with at least [filename, ?, tag]. 'x'=>0, otherwise=>1.
    :param output_csv: Where to save the final classification results (default: tir_labeled_predictions.csv).
    :param threshold_file: Where to save the best threshold.
    """
    # 1) load images & labels
    images, labels = load_images_and_labels(images_dir, csv_path)
    if not images:
        print("No images were loaded. Aborting.")
        return

    # 2) compute entropies
    entropies = []
    for img_path in images:
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"Warning: Unable to read '{img_path}'. Skipping.")
            entropies.append(None)
        else:
            entropies.append(compute_entropy(img_gray))

    # filter out any None entropies
    valid_data = [(img, lab, ent) for img, lab, ent in zip(images, labels, entropies) if ent is not None]
    if not valid_data:
        print("No valid images with computed entropies. Aborting.")
        return

    images, labels, entropies = zip(*valid_data)

    # 3) best threshold
    best_thresh, sens, spec = find_best_threshold(entropies, labels, num_thresholds=100)
    if best_thresh is None:
        print("Could not find a best threshold. Aborting.")
        return

    print(f"Best Threshold: {best_thresh:.3f}")
    print(f"Sensitivity: {sens:.3f}")
    print(f"Specificity: {spec:.3f}")

    # 4) save threshold
    joblib.dump(best_thresh, threshold_file)
    print(f"Threshold saved to: {threshold_file}")

    # 5) predict and save results
    predictions = [1 if e > best_thresh else 0 for e in entropies]
    results_df = pd.DataFrame({
        "image": images,
        "label": labels,
        "entropy": entropies,
        "predicted_label": predictions
    })
    results_df.to_csv(output_csv, index=False)
    print(f"Classification results saved to: {output_csv}")

    # plot_entropy_distribution(entropies, labels, best_thresh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a threshold-based classifier on TIR images via entropy.")

    parser.add_argument("images_dir", type=str,
                        help="Directory containing TIR images")
    parser.add_argument("csv_path", type=str,
                        help="CSV file with labeled TIR images")

    # optional
    parser.add_argument("--output_csv", type=str, default="data/filtering/tir_labeled_predictions.csv",
                        help="Path to the CSV of classification results")
    parser.add_argument("--threshold_file", type=str, default="data/filtering/tir_entropy_threshold.joblib",
                        help="Where to save the best threshold")

    args = parser.parse_args()

    main(
        images_dir=args.images_dir,
        csv_path=args.csv_path,
        output_csv=args.output_csv,
        threshold_file=args.threshold_file
    )
