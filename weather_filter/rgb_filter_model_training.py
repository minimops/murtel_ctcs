import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

"""
Script to train and evaluate either a RandomForest or SVM model on extracted image features.

Usage:
    python script.py --train_features train_extracted_features.csv --test_features test_extracted_features.csv --model rf

"""

def main(
    train_features="data/filtering/train_extracted_features.csv",
    test_features="data/filtering/test_extracted_features.csv",
    model="svm"
):
    """
    Main function to load train/test features, select a model ('rf' or 'svm'), train it,
    and print relevant performance metrics.

    :param train_features: Path to the CSV of extracted features for training.
    :param test_features: Path to the CSV of extracted features for testing.
    :param model: String specifying which model to use ('rf' or 'svm').
    """
    # Load training and testing data
    train_df = pd.read_csv(train_features)
    test_df = pd.read_csv(test_features)

    X_train = train_df.drop(columns=["image", "binary_tag"])
    y_train = train_df["binary_tag"]

    X_test = test_df.drop(columns=["image", "binary_tag"])
    y_test = test_df["binary_tag"]

    if model.lower() == "rf":
        print("RF")

        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_train, y_train)

        # testing
        y_pred = rf_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.2f}")

        # metrics
        print("\n")
        print(classification_report(y_test, y_pred, target_names=["not_clear", "clear"]))
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("\n")
        print(conf_matrix)

        feature_importances = rf_clf.feature_importances_
        feature_importance_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": feature_importances
        }).sort_values(by="Importance", ascending=False)
        print("\n")
        print(feature_importance_df)

        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_percent = (conf_matrix / np.sum(conf_matrix)) * 100

        # confusion Matrix als DataFrame mit Achsenbeschriftungen
        conf_matrix_percent_df = pd.DataFrame(
            conf_matrix_percent,
            index=["Actual Negative (0)", "Actual Positive (1)"],
            columns=["Predicted Negative (0)", "Predicted Positive (1)"]
        )

        print("\nConfusion Matrix:")
        print(conf_matrix_percent_df)

        return rf_clf

    elif model.lower() == "svm":

        print("SVM")

        clf = SVC(kernel='rbf', gamma=2.0, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_percent = (conf_matrix / np.sum(conf_matrix)) * 100

        # confusion Matrix als DataFrame mit Achsenbeschriftungen
        conf_matrix_percent_df = pd.DataFrame(
            conf_matrix_percent,
            index=["Actual Negative (0)", "Actual Positive (1)"],
            columns=["Predicted Negative (0)", "Predicted Positive (1)"]
        )

        print("\nConfusion Matrix:")
        print(conf_matrix_percent_df)

        # metrics
        print(f"\nAccuracy: {accuracy:.2f}")
        print("\n")
        print(report)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        class_accuracy_df = pd.DataFrame({
            'Class': np.unique(y_test),
            'Accuracy': class_accuracy
        })
        print("\n")
        print(class_accuracy_df)

        # perm_importance = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42)
        #
        # # Sort feature importance values and corresponding feature names
        # sorted_idx = perm_importance.importances_mean.argsort()[::-1]
        # sorted_importances = perm_importance.importances_mean[sorted_idx]

        perm_importance = permutation_importance(clf, X_test, y_test)

        feature_names = list(X_train.columns)
        features = np.array(feature_names)

        sorted_idx = perm_importance.importances_mean.argsort()
        # plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
        print("\nFeature Importance:")
        for i in sorted_idx:
            print(features[i], perm_importance.importances_mean[i])

        return clf

    else:
        raise ValueError("Model must be either 'rf' or 'svm'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate either a RandomForest or SVM model on extracted features."
    )
    parser.add_argument(
        "--train_features",
        type=str,
        default="data/filtering/train_extracted_features.csv",
        help="Path to the training features CSV (default: RF2/train_extracted_features.csv)."
    )
    parser.add_argument(
        "--test_features",
        type=str,
        default="data/filtering/test_extracted_features.csv",
        help="Path to the testing features CSV (default: RF2/test_extracted_features.csv)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="svm",
        help="Which model to use ('rf' or 'svm'). Default is 'rf'."
    )

    args = parser.parse_args()
    main(
        train_features=args.train_features,
        test_features=args.test_features,
        model=args.model
    )
