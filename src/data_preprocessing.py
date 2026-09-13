"""
data_preprocessing.py

Preprocess the 22-feature phishing URL dataset
before Random Forest training.

INPUT:
    data/phishing_22_features.csv

OUTPUT:
    data/X_train.csv
    data/X_test.csv
    data/y_train.csv
    data/y_test.csv

LABEL CONVENTION
----------------
Original PhiUSIIL dataset:
    0 = Phishing
    1 = Legitimate

Our project keeps the same convention:
    0 = Phishing
    1 = Legitimate

IMPORTANT
---------
Random Forest does NOT require feature scaling.

Therefore, StandardScaler / MinMaxScaler is intentionally
not used in this preprocessing pipeline.
"""


import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from feature_extractor import ALL_FEATURES


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIRECTORY = os.path.join(PROJECT_ROOT, "data")

INPUT_DATASET = os.path.join(
    DATA_DIRECTORY,
    "phishing_22_features.csv"
)

OUTPUT_DIRECTORY = os.path.join(
    DATA_DIRECTORY,
    "processed"
)


# ------------------------------------------------------------
# Train/Test split
# ------------------------------------------------------------

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# EXPECTED COLUMNS
# ============================================================

EXPECTED_COLUMNS = (
    ["URL"]
    + ALL_FEATURES
    + ["label"]
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    """
    Load the generated 22-feature dataset.
    """

    print("\n" + "=" * 70)
    print("LOADING 22-FEATURE DATASET")
    print("=" * 70)

    if not os.path.exists(
        INPUT_DATASET
    ):

        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{INPUT_DATASET}\n\n"
            f"Run generate_training_dataset.py first."
        )

    df = pd.read_csv(
        INPUT_DATASET
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    return df


# ============================================================
# VALIDATE COLUMN STRUCTURE
# ============================================================

def validate_columns(df):
    """
    Make sure the dataset contains exactly:

        URL
        22 features
        label

    in the correct order.
    """

    print("\n" + "=" * 70)
    print("VALIDATING DATASET STRUCTURE")
    print("=" * 70)

    actual_columns = list(
        df.columns
    )

    if actual_columns != EXPECTED_COLUMNS:

        print("\nExpected columns:")

        for number, column in enumerate(
            EXPECTED_COLUMNS,
            start=1
        ):

            print(
                f"{number:02d}. {column}"
            )

        print("\nActual columns:")

        for number, column in enumerate(
            actual_columns,
            start=1
        ):

            print(
                f"{number:02d}. {column}"
            )

        raise ValueError(
            "\nDataset column structure is incorrect."
        )

    if len(df.columns) != 24:

        raise ValueError(
            f"Expected 24 columns "
            f"(URL + 22 features + label), "
            f"got {len(df.columns)}."
        )

    print(
        "✓ 24 columns found."
    )

    print(
        "✓ Feature names are correct."
    )

    print(
        "✓ Feature order is correct."
    )


# ============================================================
# REMOVE INVALID URL ROWS
# ============================================================

def clean_urls(df):
    """
    Clean URL column.
    """

    print("\n" + "=" * 70)
    print("CLEANING URL DATA")
    print("=" * 70)

    initial_rows = len(df)

    # --------------------------------------------------------
    # Remove missing URLs
    # --------------------------------------------------------

    df = df.dropna(
        subset=["URL"]
    ).copy()

    # --------------------------------------------------------
    # Convert to string
    # --------------------------------------------------------

    df["URL"] = (
        df["URL"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Remove empty URLs
    # --------------------------------------------------------

    df = df[
        df["URL"] != ""
    ]

    # --------------------------------------------------------
    # Remove duplicate URLs
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["URL"],
        keep="first"
    )

    removed = (
        initial_rows
        - len(df)
    )

    print(
        f"Rows before cleaning : "
        f"{initial_rows:,}"
    )

    print(
        f"Rows removed         : "
        f"{removed:,}"
    )

    print(
        f"Rows after cleaning  : "
        f"{len(df):,}"
    )

    return df.reset_index(
        drop=True
    )


# ============================================================
# CONVERT FEATURES TO NUMERIC
# ============================================================

def convert_features_to_numeric(df):
    """
    Convert all 22 feature columns to numeric values.
    """

    print("\n" + "=" * 70)
    print("CONVERTING FEATURES TO NUMERIC")
    print("=" * 70)

    for feature in ALL_FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    print(
        "✓ All 22 features converted to numeric."
    )

    return df


# ============================================================
# HANDLE MISSING AND INFINITE VALUES
# ============================================================

def handle_invalid_values(df):
    """
    Handle NaN and infinite values.

    Strategy:
        - Convert +inf / -inf to NaN
        - Replace missing feature values with median
    """

    print("\n" + "=" * 70)
    print("HANDLING MISSING / INFINITE VALUES")
    print("=" * 70)

    # --------------------------------------------------------
    # Convert infinity to NaN
    # --------------------------------------------------------

    df[ALL_FEATURES] = df[
        ALL_FEATURES
    ].replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    # --------------------------------------------------------
    # Count missing values
    # --------------------------------------------------------

    missing_before = (
        df[ALL_FEATURES]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing/invalid feature values: "
        f"{missing_before:,}"
    )

    # --------------------------------------------------------
    # Fill missing feature values
    #
    # Median is used because it is less sensitive to
    # extreme values than the mean.
    # --------------------------------------------------------

    for feature in ALL_FEATURES:

        if df[feature].isna().any():

            median_value = (
                df[feature]
                .median()
            )

            # If an entire feature somehow contains
            # missing values, use 0 as a safe fallback.
            if pd.isna(
                median_value
            ):

                median_value = 0

            df[feature] = (
                df[feature]
                .fillna(median_value)
            )

    missing_after = (
        df[ALL_FEATURES]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing values after processing: "
        f"{missing_after:,}"
    )

    if missing_after != 0:

        raise ValueError(
            "Missing values still exist after preprocessing."
        )

    print(
        "✓ Missing/infinite values handled."
    )

    return df


# ============================================================
# VALIDATE LABEL
# ============================================================

def validate_label(df):
    """
    Validate the target label.

    PhiUSIIL:
        0 = phishing
        1 = legitimate
    """

    print("\n" + "=" * 70)
    print("VALIDATING TARGET LABEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Convert label to numeric
    # --------------------------------------------------------

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove rows with missing labels
    # --------------------------------------------------------

    missing_labels = (
        df["label"]
        .isna()
        .sum()
    )

    if missing_labels > 0:

        print(
            f"Removing {missing_labels} "
            f"rows with missing labels."
        )

        df = df.dropna(
            subset=["label"]
        ).copy()

    # --------------------------------------------------------
    # Convert to integer
    # --------------------------------------------------------

    df["label"] = (
        df["label"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Check valid labels
    # --------------------------------------------------------

    unique_labels = set(
        df["label"].unique()
    )

    if not unique_labels.issubset(
        {0, 1}
    ):

        raise ValueError(
            f"Invalid labels found: "
            f"{unique_labels}"
        )

    if len(unique_labels) != 2:

        raise ValueError(
            "Dataset must contain both "
            "phishing (0) and legitimate (1) classes."
        )

    print(
        "✓ Labels are valid."
    )

    print(
        "✓ 0 = Phishing"
    )

    print(
        "✓ 1 = Legitimate"
    )

    return df


# ============================================================
# VALIDATE FEATURE VALUES
# ============================================================

def validate_feature_values(df):
    """
    Check that all 22 features contain valid numeric values.
    """

    print("\n" + "=" * 70)
    print("VALIDATING 22 FEATURES")
    print("=" * 70)

    for feature in ALL_FEATURES:

        # ----------------------------------------------------
        # Numeric check
        # ----------------------------------------------------

        if not pd.api.types.is_numeric_dtype(
            df[feature]
        ):

            raise ValueError(
                f"Feature '{feature}' "
                f"is not numeric."
            )

        # ----------------------------------------------------
        # NaN check
        # ----------------------------------------------------

        if df[feature].isna().any():

            raise ValueError(
                f"Feature '{feature}' "
                f"contains NaN values."
            )

        # ----------------------------------------------------
        # Infinite check
        # ----------------------------------------------------

        if np.isinf(
            df[feature].to_numpy()
        ).any():

            raise ValueError(
                f"Feature '{feature}' "
                f"contains infinite values."
            )

    print(
        "✓ All 22 features are numeric."
    )

    print(
        "✓ No NaN values."
    )

    print(
        "✓ No infinite values."
    )


# ============================================================
# DISPLAY CLASS DISTRIBUTION
# ============================================================

def display_class_distribution(
    df,
    title
):
    """
    Display class distribution.
    """

    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)

    counts = (
        df["label"]
        .value_counts()
        .sort_index()
    )

    phishing = counts.get(
        0,
        0
    )

    legitimate = counts.get(
        1,
        0
    )

    total = len(df)

    print(
        f"Phishing   (0): "
        f"{phishing:,}"
    )

    print(
        f"Legitimate (1): "
        f"{legitimate:,}"
    )

    if total > 0:

        print(
            f"Phishing percentage   : "
            f"{phishing / total * 100:.2f}%"
        )

        print(
            f"Legitimate percentage : "
            f"{legitimate / total * 100:.2f}%"
        )


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

def separate_features_and_target(df):
    """
    Separate X and y.

    URL is deliberately excluded from X.

    X:
        22 numerical features

    y:
        label
    """

    print("\n" + "=" * 70)
    print("SEPARATING FEATURES AND TARGET")
    print("=" * 70)

    X = df[
        ALL_FEATURES
    ].copy()

    y = df[
        "label"
    ].copy()

    print(
        f"X shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )

    print(
        "\nURL column excluded from model features."
    )

    print(
        "✓ X contains exactly 22 features."
    )

    return X, y


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_dataset(X, y):
    """
    Split dataset into training and testing sets.

    Stratification preserves the phishing/legitimate
    class distribution in both sets.
    """

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    print(
        f"Training samples : "
        f"{len(X_train):,}"
    )

    print(
        f"Testing samples  : "
        f"{len(X_test):,}"
    )

    print(
        f"Test size        : "
        f"{TEST_SIZE * 100:.0f}%"
    )

    print(
        "\n✓ Stratified split completed."
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# VALIDATE TRAIN / TEST SPLIT
# ============================================================

def validate_split(
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Validate the train/test datasets.
    """

    print("\n" + "=" * 70)
    print("VALIDATING TRAIN / TEST DATA")
    print("=" * 70)

    # --------------------------------------------------------
    # Feature count
    # --------------------------------------------------------

    if X_train.shape[1] != 22:

        raise ValueError(
            "X_train does not contain exactly 22 features."
        )

    if X_test.shape[1] != 22:

        raise ValueError(
            "X_test does not contain exactly 22 features."
        )

    # --------------------------------------------------------
    # Same feature order
    # --------------------------------------------------------

    if list(X_train.columns) != list(
        X_test.columns
    ):

        raise ValueError(
            "Feature order differs between train and test."
        )

    if list(X_train.columns) != ALL_FEATURES:

        raise ValueError(
            "Training feature order does not match "
            "feature_extractor.py."
        )

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if len(X_train) != len(y_train):

        raise ValueError(
            "X_train and y_train lengths do not match."
        )

    if len(X_test) != len(y_test):

        raise ValueError(
            "X_test and y_test lengths do not match."
        )

    # --------------------------------------------------------
    # Check NaN
    # --------------------------------------------------------

    if X_train.isna().any().any():

        raise ValueError(
            "NaN values found in X_train."
        )

    if X_test.isna().any().any():

        raise ValueError(
            "NaN values found in X_test."
        )

    # --------------------------------------------------------
    # Check classes
    # --------------------------------------------------------

    if set(y_train.unique()) != {0, 1}:

        raise ValueError(
            "Training set does not contain both classes."
        )

    if set(y_test.unique()) != {0, 1}:

        raise ValueError(
            "Testing set does not contain both classes."
        )

    print(
        "✓ Feature count: PASS"
    )

    print(
        "✓ Feature order: PASS"
    )

    print(
        "✓ X/y dimensions: PASS"
    )

    print(
        "✓ Missing values: PASS"
    )

    print(
        "✓ Both classes present: PASS"
    )

    # --------------------------------------------------------
    # Display distributions
    # --------------------------------------------------------

    train_distribution = (
        y_train
        .value_counts()
        .sort_index()
    )

    test_distribution = (
        y_test
        .value_counts()
        .sort_index()
    )

    print("\nTraining class distribution:")

    print(
        f"Phishing   (0): "
        f"{train_distribution.get(0, 0):,}"
    )

    print(
        f"Legitimate (1): "
        f"{train_distribution.get(1, 0):,}"
    )

    print("\nTesting class distribution:")

    print(
        f"Phishing   (0): "
        f"{test_distribution.get(0, 0):,}"
    )

    print(
        f"Legitimate (1): "
        f"{test_distribution.get(1, 0):,}"
    )

    print(
        "\n✓ TRAIN / TEST VALIDATION PASSED."
    )


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Save processed train/test datasets.
    """

    print("\n" + "=" * 70)
    print("SAVING PROCESSED DATA")
    print("=" * 70)

    X_train_path = os.path.join(
        OUTPUT_DIRECTORY,
        "X_train.csv"
    )

    X_test_path = os.path.join(
        OUTPUT_DIRECTORY,
        "X_test.csv"
    )

    y_train_path = os.path.join(
        OUTPUT_DIRECTORY,
        "y_train.csv"
    )

    y_test_path = os.path.join(
        OUTPUT_DIRECTORY,
        "y_test.csv"
    )

    # --------------------------------------------------------
    # Save X
    # --------------------------------------------------------

    X_train.to_csv(
        X_train_path,
        index=False
    )

    X_test.to_csv(
        X_test_path,
        index=False
    )

    # --------------------------------------------------------
    # Save y
    # --------------------------------------------------------

    y_train.to_csv(
        y_train_path,
        index=False
    )

    y_test.to_csv(
        y_test_path,
        index=False
    )

    print(
        f"\n✓ {X_train_path}"
    )

    print(
        f"✓ {X_test_path}"
    )

    print(
        f"✓ {y_train_path}"
    )

    print(
        f"✓ {y_test_path}"
    )


# ============================================================
# DISPLAY FINAL SUMMARY
# ============================================================

def display_final_summary(
    df,
    X_train,
    X_test
):
    """
    Display preprocessing summary.
    """

    print("\n" + "=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)

    print(
        f"\nOriginal/generated rows : "
        f"{len(df):,}"
    )

    print(
        f"Training rows           : "
        f"{len(X_train):,}"
    )

    print(
        f"Testing rows            : "
        f"{len(X_test):,}"
    )

    print(
        f"Number of features      : "
        f"{len(ALL_FEATURES)}"
    )

    print(
        f"Test size               : "
        f"{TEST_SIZE * 100:.0f}%"
    )

    print(
        f"Random state            : "
        f"{RANDOM_STATE}"
    )

    print(
        "\nFeature scaling:"
    )

    print(
        "Not applied — Random Forest "
        "does not require scaling."
    )

    print(
        "\nLabel:"
    )

    print(
        "0 = Phishing"
    )

    print(
        "1 = Legitimate"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("AI PHISHING DETECTION")
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 70)

    # ========================================================
    # STEP 1
    # Load dataset
    # ========================================================

    df = load_dataset()

    # ========================================================
    # STEP 2
    # Validate columns
    # ========================================================

    validate_columns(
        df
    )

    # ========================================================
    # STEP 3
    # Clean URLs
    # ========================================================

    df = clean_urls(
        df
    )

    # ========================================================
    # STEP 4
    # Convert features to numeric
    # ========================================================

    df = convert_features_to_numeric(
        df
    )

    # ========================================================
    # STEP 5
    # Validate label
    # ========================================================

    df = validate_label(
        df
    )

    # ========================================================
    # STEP 6
    # Handle NaN/infinite values
    # ========================================================

    df = handle_invalid_values(
        df
    )

    # ========================================================
    # STEP 7
    # Validate all features
    # ========================================================

    validate_feature_values(
        df
    )

    # ========================================================
    # STEP 8
    # Display class distribution
    # ========================================================

    display_class_distribution(
        df,
        "CLEAN DATASET CLASS DISTRIBUTION"
    )

    # ========================================================
    # STEP 9
    # Separate X and y
    # ========================================================

    X, y = separate_features_and_target(
        df
    )

    # ========================================================
    # STEP 10
    # Train/test split
    # ========================================================

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_dataset(
        X,
        y
    )

    # ========================================================
    # STEP 11
    # Validate split
    # ========================================================

    validate_split(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ========================================================
    # STEP 12
    # Save processed data
    # ========================================================

    save_processed_data(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ========================================================
    # STEP 13
    # Final summary
    # ========================================================

    display_final_summary(
        df,
        X_train,
        X_test
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("DATA PREPROCESSING COMPLETE ✓")
    print("=" * 70)

    print(
        "\nThe processed datasets are ready "
        "for Random Forest training."
    )

    print(
        "\nNext file:"
    )

    print(
        "    train_model.py"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()