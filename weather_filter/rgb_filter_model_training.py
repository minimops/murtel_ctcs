import argparse
from datetime import datetime
import joblib
import os

from extraction import main as extract_features
from train_test_splitting import main as train_test_splitting
from rgb_filter_models import main as rgb_filter_model_training

"""
Script that orchestrates:
1) Feature extraction (extraction.py),
2) Train-test splitting (train_test_splitting.py),
3) Model training (rgb_filter_models.py),
4) Saving the trained model with joblib.

Usage example (all arguments optional except for img_dir and csv_file):
    python rgb_filter_model_training.py \
        data/RGB_DL_images \
        data/filtering/labeled_images.csv \
        --extracted_features data/filtering/all_extracted_features.csv \
        --train_features data/filtering/train_extracted_features.csv \
        --test_features data/filtering/test_extracted_features.csv \
        --train_csv data/filtering/train.csv \
        --test_csv data/filtering/test.csv \
        --model svm \
        --model_dir data/filtering \
        --re_extract
"""

def main(img_dir,
        csv_file,
        extracted_features="data/filtering/all_extracted_features.csv",
        train_features="data/filtering/train_extracted_features.csv",
        test_features="data/filtering/test_extracted_features.csv",
        train_csv="data/filtering/train.csv",
        test_csv="data/filtering/test.csv",
        model="svm",
        model_dir="data/filtering",
        re_extract=False):
    """
    Main function that runs the pipeline:
      1) Feature extraction (skipped if extracted_features exists and re_extract=False),
      2) Train-test split,
      3) Model training,
      4) Saving the model via joblib.

    :param img_dir: Directory containing the images for feature extraction.
    :param csv_file: CSV file containing all labeled images.
    :param extracted_features: Where to save (or load) all extracted features.
    :param train_features: Where to save extracted training features after splitting.
    :param test_features: Where to save extracted testing features after splitting.
    :param train_csv: Where to save the train subset CSV (with minimal info).
    :param test_csv: Where to save the test subset CSV (with minimal info).
    :param model: 'rf' or 'svm' to specify which model to train.
    :param model_dir: Where to save the trained model.
    :param re_extract: Boolean flag to force re-extraction of features if True.
    """

    print("\n=== RGB WEATHER FILTERING ===\n")
    timestamp = datetime.now().strftime("%d%m%H%M")

    # 1) Feature Extraction
    if os.path.exists(extracted_features) and not re_extract:
        print(f"\n=== Feature extraction skipped. Found existing file: {extracted_features}.\n"
              f"    Use --re_extract to force re-extraction.")
    else:
        print("\n=== Feature Extraction ===")
        scaler = extract_features(
                    img_dir=img_dir,
                    all_images_csv=csv_file,
                    output_features=extracted_features
                )
        # 4) Saving the scaler
        print("\n=== Saving Scaler ===")
        scaler_path = os.path.join(model_dir, f'scaler_{timestamp}.joblib')
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")

    # 2) Train-Test Splitting
    print("\n=== Train-Test Splitting ===")
    train_test_splitting(
        all_features_csv=extracted_features,
        train_features_csv=train_features,
        test_features_csv=test_features,
        train_csv=train_csv,
        test_csv=test_csv,
        test_size=0.2
    )

    # 3) Model Training
    print("\n=== Model Training ===")
    trained_model = rgb_filter_model_training(
        train_features=train_features,
        test_features=test_features,
        model=model
    )

    # 4) Saving the Model
    print("\n=== Saving Model ===")
    model_path = os.path.join(model_dir, f'{model}_model_{timestamp}.joblib')
    joblib.dump(trained_model, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        "img_dir",
        type=str,
        help="Directory containing images for feature extraction."
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="CSV file with labeled images."
    )

    # Optional arguments
    parser.add_argument(
        "--extracted_features",
        type=str,
        default="data/filtering/all_extracted_features.csv",
        help="Path for the combined features CSV (default: data/filtering/all_extracted_features.csv)."
    )
    parser.add_argument(
        "--train_features",
        type=str,
        default="data/filtering/train_extracted_features.csv",
        help="Path for the extracted training features CSV."
    )
    parser.add_argument(
        "--test_features",
        type=str,
        default="data/filtering/test_extracted_features.csv",
        help="Path for the extracted testing features CSV."
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="data/filtering/train.csv",
        help="Where to save the train subset CSV with minimal info (default: data/filtering/train.csv)."
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="data/filtering/test.csv",
        help="Where to save the test subset CSV with minimal info (default: data/filtering/test.csv)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="svm",
        help="Choose which model to train: 'rf' or 'svm' (default: svm)."
    )
    parser.add_argument(
        "--re_extract",
        action="store_true",
        default=False,
        help="If set, forces re-extraction of features even if an existing file is found."
    )

    args = parser.parse_args()

    main(
        img_dir=args.img_dir,
        csv_file=args.csv_file,
        extracted_features=args.extracted_features,
        train_features=args.train_features,
        test_features=args.test_features,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        model=args.model,
        re_extract=args.re_extract
    )
