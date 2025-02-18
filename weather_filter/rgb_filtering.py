import argparse
import os
import tempfile
from tempfile import tempdir

import pandas as pd
import joblib
from extraction import extract_and_save_all_features

"""
Script that:
1. Extracts features for a new CSV of unlabeled images (using extraction.py).
2. Loads a trained model file.
3. Predicts labels for these images.
4. Saves a CSV with [image, predicted_label].

Usage example:
    python rgb_filtering.py \
        data/RGB_DL_images \
        --scaler_path scaler.joblib \
        --model_path model.joblib \
        --output_features data/filtering/predictions.csv
"""

def main(img_dir,
        output_features="data/filtering/predict_extracted_features.csv",
        model_path="svm_model.joblib",
        scaler_path="scaler.joblib",
        output_predictions="predictions.csv"):
    """
    :param img_dir: Directory with the unlabeled images.
    :param output_features: Where to store the extracted feature CSV for these images.
    :param model_path: Path to the trained model
    :param scaler_path: Path to the scaler
    :param output_predictions: Where to save predictions with columns [image, predicted_label].
    """

    # 1) gather all image filenames
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    all_files = os.listdir(img_dir)
    image_files = [f for f in all_files if f.lower().endswith(valid_exts)]
    if not image_files:
        print(f"No images found in directory: {img_dir}")
        return

    print(f"Found {len(image_files)} images in {img_dir}.\n")

    # 2) build a temporary CSV so extraction.py can parse them
    temp_csv = os.path.join(tempfile.gettempdir(), "pred_temp_fnames.csv")
    print(f"Building temporary CSV at {temp_csv}")
    pd.DataFrame({"image": image_files}).to_csv(temp_csv, index=False)

    # 3) extract features
    print(f"=== Extracting features for unlabeled images ===")
    # load scaler
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    print(f"Loaded scaler from {scaler_path}\n")

    extract_and_save_all_features(
        csv_file=temp_csv,
        img_dir=img_dir,
        output_path=output_features,
        model_scaler=scaler
    )
    os.remove(temp_csv)

    # 4) load the trained model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    print(f"Loaded trained model from {model_path}\n")

    # 5) prediction
    feat_df = pd.read_csv(output_features)

    images = feat_df["image"].values
    drop_cols = ["image", "binary_tag"] if "binary_tag" in feat_df.columns else ["image"]
    X = feat_df.drop(columns=drop_cols, errors="ignore")

    y_pred = model.predict(X)

    # 6) save predictions ; 0=not_clear, 1=clear.
    results_df = pd.DataFrame({"image": images, "predicted_label": y_pred})
    label_map = {0: "not_clear", 1: "clear"}
    if set(y_pred) <= {0, 1}:
        results_df["predicted_tag"] = results_df["predicted_label"].map(label_map)

    results_df.to_csv(output_predictions, index=False)
    print(f"Predictions saved to {output_predictions}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "img_dir",
        type=str,
        help="Directory containing the unlabeled images."
    )
    parser.add_argument(
        "--scaler_path",
        type=str,
        required=True,
        help="Path to the trained scaler file."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model file."
    )

    # optional args
    parser.add_argument(
        "--output_features",
        type=str,
        default="data/filtering/predict_extracted_features.csv",
        help="Where to save the extracted features CSV."
    )

    parser.add_argument(
        "--output_predictions",
        type=str,
        default="predictions.csv",
        help="Where to save the prediction CSV."
    )

    args = parser.parse_args()

    main(
        img_dir=args.img_dir,
        output_features=args.output_features,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        output_predictions=args.output_predictions
    )
