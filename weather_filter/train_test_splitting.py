import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

"""
Script to split an extracted features into train and test subsets.

Usage:
    python train_test_splitting.py data/filtering/all_extracted_features.csv

Outputs train_extracted_features.csv and test_extracted_features.csv 
as well as train.csv and test.csv.

"""

def main(all_features_csv="data/filtering/all_extracted_features.csv",
         train_features_csv="data/filtering/train_extracted_features.csv",
         test_features_csv="data/filtering/test_extracted_features.csv",
         train_csv="data/filtering/train.csv",
         test_csv="data/filtering/test.csv",
         test_size=0.2, random_state=42):

    """
    Reads the all_features_csv, does train_test_split, saves subsets:
      1) train_extracted_features.csv and train.csv
      2) test_extracted_features.csv and test.csv
    """

    all_df = pd.read_csv(all_features_csv)
    train_df, test_df = train_test_split(
        all_df,
        test_size=test_size,
        random_state=random_state,
        stratify=all_df["binary_tag"]
    )

    minimal_train = train_df[["image", "binary_tag"]]
    minimal_train.to_csv(train_csv, index=False)
    print(f"Train labels CSV saved to: {train_csv}")
    train_df.to_csv(train_features_csv, index=False)
    print(f"Train feature CSV saved to: {train_features_csv}")

    minimal_test = test_df[["image", "binary_tag"]]
    minimal_test.to_csv(test_csv, index=False)
    print(f"Test labels CSV saved to: {test_csv}")
    test_df.to_csv(test_features_csv, index=False)
    print(f"Test feature CSV saved to: {test_features_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "all_features_csv",
        type=str,
        default="data/filtering/all_extracted_features.csv",
        help="Path to the combined features CSV (default: data/filtering/all_extracted_features.csv)."
    )

    parser.add_argument(
        "--train_features_csv",
        type=str,
        default="data/filtering/train_extracted_features.csv",
        help="Output path for the extracted features (train subset)."
    )
    parser.add_argument(
        "--test_features_csv",
        type=str,
        default="data/filtering/test_extracted_features.csv",
        help="Output path for the extracted features (test subset)."
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="data/filtering/train.csv",
        help="Output path for minimal train CSV with [image,binary_tag]."
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="data/filtering/test.csv",
        help="Output path for minimal test CSV with [image,binary_tag]."
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data to use for test set (default: 0.2)."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for the split (default: 42)."
    )

    args = parser.parse_args()

    main(
        all_features_csv=args.all_features_csv,
        train_features_csv=args.train_features_csv,
        test_features_csv=args.test_features_csv,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        test_size=args.test_size,
        random_state=args.random_state
    )
