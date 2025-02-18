import argparse
import os
import joblib
import cv2
import pandas as pd

from tir_filter_train import compute_entropy

"""
Script that:
 - calculates entropies of unlabeled TIR images
 - classifies them

Usage example:
    python tir_filtering.py \
        data/TIR_images \
        data/filtering/tir_entropy_threshold.joblib
"""

def main(images_dir,
        threshold_file="tir_entropy_threshold.joblib",
        output_csv="tir_unlabeled_predictions.csv"):
    """
    Classifies unlabeled TIR images via a pre-trained threshold.
    """

    # 1) load threshold
    if not os.path.isfile(threshold_file):
        raise FileNotFoundError(f"Threshold file not found: {threshold_file}")
    threshold = joblib.load(threshold_file)
    print(f"Loaded threshold: {threshold:.3f}")

    # 2) get images
    valid_exts = (".jpg", ".jpeg", ".png")
    all_files = os.listdir(images_dir)
    image_files = [f for f in all_files if f.lower().endswith(valid_exts)]
    if not image_files:
        print(f"No images found in directory: {images_dir}")
        return

    # 3) calc entropy & predict
    predictions_list = []
    for filename in image_files:
        img_path = os.path.join(images_dir, filename)
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"could not read '{img_path}'. Skipping.")
            continue

        entropy_val = compute_entropy(img_gray)
        predicted_label = 1 if entropy_val > threshold else 0
        predictions_list.append((filename, entropy_val, predicted_label))

    # 4) save to csv
    results_df = pd.DataFrame(predictions_list, columns=["image", "entropy", "predicted_label"])
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "images_dir",
        type=str,
        help="Directory containing unlabeled TIR images (default: ../../RGB_DL_images)."
    )
    parser.add_argument(
        "threshold_file",
        type=str,
        help="Path to the saved threshold (default: tir_entropy_threshold.joblib)."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/filtering/tir_unlabeled_predictions.csv",
        help="Output CSV for predictions."
    )

    args = parser.parse_args()
    main(
        images_dir=args.images_dir,
        threshold_file=args.threshold_file,
        output_csv=args.output_csv
    )
