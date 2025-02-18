import os
import cv2
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from features import extract_features

"""
Script to extract image features for ALL labeled images at once.

Usage:
    python extraction.py /path/to/images data/filtering/all_images.csv data/filtering/all_extracted_features.csv
"""

def process_img_features(img_path):
    image = cv2.imread(img_path)
    features = extract_features(image)
    return features


def to_binary_labels(df):
    """
    Converts string labels ('c' vs. others) into binary values (1 or 0)
    """
    df['binary_tag'] = df['tag'].apply(lambda x: 1 if x == 'c' else 0)
    return df


def parse_time_from_filename(filename):
    yy = int(filename[1:3])  # year
    mm = int(filename[3:5])  # month
    hh = int(filename[7:9])  # hour

    if mm in [12, 1, 2]:
        season = "Winter"
    elif mm in [3, 4, 5]:
        season = "Spring"
    elif mm in [6, 7, 8]:
        season = "Summer"
    else:
        season = "Fall"

    if 6 <= hh < 12:
        time_of_day = "Morning"
    elif 12 <= hh < 18:
        time_of_day = "Afternoon"
    elif 18 <= hh < 24:
        time_of_day = "Evening"
    else:
        time_of_day = "Night"

    return season, time_of_day


def extract_and_save_all_features(csv_file, img_dir, output_path):
    """
    Reads a CSV file referencing ALL labeled images, extracts features for each image,
    and saves a single all-encompassing feature CSV.
    """
    df = pd.read_csv(csv_file)
    df = to_binary_labels(df)

    features_list = []
    for _, row in df.iterrows():
        filename = row['image']
        img_path = os.path.join(img_dir, filename)
        if not os.path.isfile(img_path):
            print(f"File not found (skipped): {img_path}")
            continue

        print(f"Processing Image: {filename}")
        feats = process_img_features(img_path)
        if feats:
            season, time_of_day = parse_time_from_filename(filename)
            feats["season"] = season
            feats["time_of_day"] = time_of_day
            feats["image"] = filename
            feats["binary_tag"] = row["binary_tag"]
            features_list.append(feats)

    features_df = pd.DataFrame(features_list)

    # one-hot encode
    features_df = pd.get_dummies(features_df, columns=["season", "time_of_day"])

    # scale numeric columns except:
    #   - 'binary_tag', 'image'
    #   - and any one-hot columns (start with "season_", "time_of_day_")
    exclude_cols = ["binary_tag", "image"]
    exclude_cols += [c for c in features_df.columns if c.startswith("season_") or c.startswith("time_of_day_")]
    numeric_cols = [c for c in features_df.columns if c not in exclude_cols]

    scaler = StandardScaler()
    features_df[numeric_cols] = scaler.fit_transform(features_df[numeric_cols])

    features_df.to_csv(output_path, index=False)
    print(f"Extracted features saved to: {output_path}")


def main(img_dir, all_images_csv="data/filtering/all_images.csv",
         output_features="data/filtering/all_extracted_features.csv"):
    extract_and_save_all_features(all_images_csv, img_dir, output_features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("img_dir", type=str, help="Directory containing images.")
    parser.add_argument(
        "--all_images_csv",
        type=str,
        default="data/filtering/all_images.csv",
        help="CSV referencing all labeled images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/filtering/all_extracted_features.csv",
        help="Where to write the single, combined features CSV"
    )

    args = parser.parse_args()
    main(
        img_dir=args.img_dir,
        all_images_csv=args.all_images_csv,
        output_features=args.output
    )
