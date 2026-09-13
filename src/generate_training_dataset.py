"""
generate_training_dataset.py

Purpose:
1. Read the original PhiUSIIL Phishing URL Dataset.
2. Keep only URL and label.
3. Save them as raw_url.csv.
4. Read raw_url.csv.
5. Extract 22 features using feature_extractor.py.
6. Save the final dataset containing:
      URL + 22 extracted features + label

Project:
AI Phishing Detection
"""

import os
import pandas as pd
import numpy as np

from feature_extractor import extract_features, ALL_FEATURES


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

INPUT_DATASET = os.path.join(DATA_DIR, "PhiUSIIL_Phishing_URL_Dataset.csv")

RAW_URL_DATASET = os.path.join(DATA_DIR, "raw_url.csv")

FINAL_DATASET = os.path.join(DATA_DIR, "phishing_22_features.csv")

# Process a manageable balanced subset for feature extraction.
MAX_URLS = 5000

# Keep the original class distribution.
# True = select an equal number of phishing and legitimate URLs.
BALANCED_DATASET = True

RANDOM_STATE = 42

# Save progress after every N URLs.
CHECKPOINT_INTERVAL = 100

CHECKPOINT_FILE = os.path.join(
    DATA_DIR,
    "phishing_22_features_checkpoint.csv"
)


# ============================================================
# EXPECTED COLUMNS
# ============================================================

FINAL_COLUMNS = ["URL"] + ALL_FEATURES + ["label"]

CHECKPOINT_COLUMNS = ["URL"] + ALL_FEATURES + ["label", "status"]


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():
    """Create required project directories."""

    os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# STEP 1: LOAD PHIUSIIL DATASET
# ============================================================

def load_phiusill_dataset():
    """
    Load the original PhiUSIIL dataset.

    Returns:
        pandas.DataFrame
    """

    if not os.path.exists(INPUT_DATASET):
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_DATASET}"
        )

    print("\nLoading PhiUSIIL dataset...")

    df = pd.read_csv(INPUT_DATASET)

    print(f"Original dataset shape: {df.shape}")

    # --------------------------------------------------------
    # Find URL column
    # --------------------------------------------------------

    url_candidates = [
        "URL",
        "url",
        "Url"
    ]

    url_column = None

    for column in url_candidates:
        if column in df.columns:
            url_column = column
            break

    if url_column is None:
        raise ValueError(
            "Could not find URL column in PhiUSIIL dataset."
        )

    # --------------------------------------------------------
    # Find label column
    # --------------------------------------------------------

    label_candidates = [
        "label",
        "Label",
        "LABEL"
    ]

    label_column = None

    for column in label_candidates:
        if column in df.columns:
            label_column = column
            break

    if label_column is None:
        raise ValueError(
            "Could not find label column in PhiUSIIL dataset."
        )

    print(f"URL column   : {url_column}")
    print(f"Label column : {label_column}")

    # --------------------------------------------------------
    # Keep ONLY URL and label
    # --------------------------------------------------------

    raw_df = df[[url_column, label_column]].copy()

    raw_df.columns = ["URL", "label"]

    # --------------------------------------------------------
    # Basic cleaning
    # --------------------------------------------------------

    raw_df["URL"] = raw_df["URL"].astype(str).str.strip()

    raw_df = raw_df[
        raw_df["URL"].notna()
        & (raw_df["URL"] != "")
        & (raw_df["URL"].str.lower() != "nan")
    ]

    # Convert label to numeric
    raw_df["label"] = pd.to_numeric(
        raw_df["label"],
        errors="coerce"
    )

    # Remove invalid labels
    raw_df = raw_df[
        raw_df["label"].isin([0, 1])
    ]

    # Remove duplicate URLs
    raw_df = raw_df.drop_duplicates(
        subset=["URL"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    if not set(raw_df["label"].unique()).issubset({0, 1}):
        raise ValueError(
            "Dataset contains labels other than 0 and 1."
        )

    if raw_df.empty:
        raise ValueError(
            "No valid URL records found."
        )

    print(f"Clean URL dataset shape: {raw_df.shape}")

    print("\nLabel distribution:")
    print(raw_df["label"].value_counts().sort_index())

    return raw_df


# ============================================================
# STEP 2: CREATE RAW URL DATASET
# ============================================================

def create_raw_url_dataset(df):
    """
    Create raw_url.csv containing only:

        URL
        label
    """

    print("\n" + "=" * 60)
    print("STEP 1: Creating raw URL dataset")
    print("=" * 60)

    raw_df = df.copy()

    # --------------------------------------------------------
    # Optional dataset size limit
    # --------------------------------------------------------

    if MAX_URLS is not None:

        if MAX_URLS <= 0:
            raise ValueError(
                "MAX_URLS must be greater than 0."
            )

        if BALANCED_DATASET:

            samples_per_class = MAX_URLS // 2

            phishing = raw_df[
                raw_df["label"] == 0
            ].sample(
                n=min(
                    samples_per_class,
                    len(raw_df[raw_df["label"] == 0])
                ),
                random_state=RANDOM_STATE
            )

            legitimate = raw_df[
                raw_df["label"] == 1
            ].sample(
                n=min(
                    samples_per_class,
                    len(raw_df[raw_df["label"] == 1])
                ),
                random_state=RANDOM_STATE
            )

            raw_df = pd.concat(
                [phishing, legitimate],
                ignore_index=True
            )

            raw_df = raw_df.sample(
                frac=1,
                random_state=RANDOM_STATE
            ).reset_index(drop=True)

        else:

            raw_df = raw_df.sample(
                n=min(MAX_URLS, len(raw_df)),
                random_state=RANDOM_STATE
            ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save raw URL dataset
    # --------------------------------------------------------

    raw_df.to_csv(
        RAW_URL_DATASET,
        index=False
    )

    print(f"\nRaw dataset saved:")
    print(f"  {RAW_URL_DATASET}")

    print(f"\nRaw dataset shape: {raw_df.shape}")

    print("\nRaw dataset columns:")
    print(list(raw_df.columns))

    print("\nRaw label distribution:")
    print(raw_df["label"].value_counts().sort_index())

    return raw_df


# ============================================================
# STEP 3: EXTRACT FEATURES
# ============================================================

def extract_features_from_raw_dataset(raw_df):
    """
    Extract 22 features from every URL in raw_url.csv.
    """

    print("\n" + "=" * 60)
    print("STEP 2: Extracting 22 features")
    print("=" * 60)

    records = []

    total_urls = len(raw_df)

    print(f"\nTotal URLs to process: {total_urls}")
    print("Feature extraction started...\n")

    for index, row in raw_df.iterrows():

        url = row["URL"]
        label = int(row["label"])

        try:

            # ------------------------------------------------
            # Extract features using feature_extractor.py
            # ------------------------------------------------

            features = extract_features(url)

            # ------------------------------------------------
            # Validate feature names
            # ------------------------------------------------

            if set(features.keys()) != set(ALL_FEATURES):

                missing = set(ALL_FEATURES) - set(features.keys())
                extra = set(features.keys()) - set(ALL_FEATURES)

                raise ValueError(
                    f"Feature mismatch. "
                    f"Missing: {missing}, Extra: {extra}"
                )

            # ------------------------------------------------
            # Create record
            # ------------------------------------------------

            record = {
                "URL": url
            }

            # Add 22 features in exact order
            for feature in ALL_FEATURES:
                record[feature] = features[feature]

            # Add original label
            record["label"] = label

            records.append(record)

        except Exception as error:

            print(
                f"[WARNING] Failed URL {index + 1}/{total_urls}: "
                f"{url}"
            )

            print(f"          Error: {error}")

        # ----------------------------------------------------
        # Progress information
        # ----------------------------------------------------

        if (index + 1) % CHECKPOINT_INTERVAL == 0:

            print(
                f"Processed: {index + 1}/{total_urls} "
                f"({((index + 1) / total_urls) * 100:.2f}%)"
            )

            # Save checkpoint
            checkpoint_df = pd.DataFrame(records)

            if not checkpoint_df.empty:
                checkpoint_df.to_csv(
                    CHECKPOINT_FILE,
                    index=False
                )

    return records


# ============================================================
# STEP 4: CREATE FINAL DATASET
# ============================================================

def create_final_dataset(records):
    """
    Create final dataset:

    URL
    + 22 extracted features
    + label
    """

    print("\n" + "=" * 60)
    print("STEP 3: Creating final feature dataset")
    print("=" * 60)

    if not records:
        raise ValueError(
            "No feature records were successfully extracted."
        )

    final_df = pd.DataFrame(records)

    # --------------------------------------------------------
    # Force exact column order
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in FINAL_COLUMNS
        if column not in final_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in final dataset: "
            f"{missing_columns}"
        )

    final_df = final_df[FINAL_COLUMNS]

    # --------------------------------------------------------
    # Convert feature columns to numeric
    # --------------------------------------------------------

    for feature in ALL_FEATURES:

        final_df[feature] = pd.to_numeric(
            final_df[feature],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Check for invalid feature values
    # --------------------------------------------------------

    feature_values = final_df[ALL_FEATURES]

    invalid_mask = (
        feature_values.isna().any(axis=1)
        |
        ~np.isfinite(feature_values).all(axis=1)
    )

    invalid_count = invalid_mask.sum()

    if invalid_count > 0:

        print(
            f"\nRemoving {invalid_count} records "
            f"with invalid feature values."
        )

        final_df = final_df[
            ~invalid_mask
        ].reset_index(drop=True)

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    final_df["label"] = pd.to_numeric(
        final_df["label"],
        errors="coerce"
    )

    final_df = final_df[
        final_df["label"].isin([0, 1])
    ].reset_index(drop=True)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if final_df.empty:
        raise ValueError(
            "Final dataset is empty."
        )

    if list(final_df.columns) != FINAL_COLUMNS:
        raise ValueError(
            "Final dataset column order is incorrect."
        )

    if final_df[ALL_FEATURES].isna().any().any():
        raise ValueError(
            "Final dataset contains NaN values."
        )

    if not np.isfinite(
        final_df[ALL_FEATURES].to_numpy()
    ).all():

        raise ValueError(
            "Final dataset contains infinite values."
        )

    # --------------------------------------------------------
    # Save final dataset
    # --------------------------------------------------------

    final_df.to_csv(
        FINAL_DATASET,
        index=False
    )

    print(f"\nFinal dataset saved:")
    print(f"  {FINAL_DATASET}")

    print(f"\nFinal dataset shape: {final_df.shape}")

    print("\nFinal dataset columns:")
    for number, column in enumerate(
        final_df.columns,
        start=1
    ):
        print(f"  {number:2}. {column}")

    print("\nFinal label distribution:")
    print(
        final_df["label"]
        .value_counts()
        .sort_index()
    )

    return final_df


# ============================================================
# STEP 5: FINAL VALIDATION
# ============================================================

def validate_final_dataset(df):
    """
    Perform strict validation of the final dataset.
    """

    print("\n" + "=" * 60)
    print("STEP 4: Final validation")
    print("=" * 60)

    # --------------------------------------------------------
    # Column validation
    # --------------------------------------------------------

    if list(df.columns) != FINAL_COLUMNS:

        raise ValueError(
            "Final dataset does not contain "
            "the expected columns/order."
        )

    # --------------------------------------------------------
    # Feature count
    # --------------------------------------------------------

    feature_count = len(ALL_FEATURES)

    if feature_count != 22:

        raise ValueError(
            f"Expected 22 features, found {feature_count}."
        )

    # --------------------------------------------------------
    # Numeric validation
    # --------------------------------------------------------

    if not all(
        pd.api.types.is_numeric_dtype(df[feature])
        for feature in ALL_FEATURES
    ):

        raise ValueError(
            "One or more feature columns are not numeric."
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    if df[ALL_FEATURES].isna().any().any():

        raise ValueError(
            "Dataset contains missing feature values."
        )

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    if not np.isfinite(
        df[ALL_FEATURES].to_numpy()
    ).all():

        raise ValueError(
            "Dataset contains infinite feature values."
        )

    # --------------------------------------------------------
    # Label validation
    # --------------------------------------------------------

    labels = set(df["label"].unique())

    if not labels.issubset({0, 1}):

        raise ValueError(
            f"Invalid labels found: {labels}"
        )

    if len(labels) < 2:

        raise ValueError(
            "Dataset must contain both classes."
        )

    print("\nValidation successful!")
    print(f"  Records       : {len(df)}")
    print(f"  URL column    : Present")
    print(f"  Label column  : Present")
    print(f"  Features      : {feature_count}")
    print(f"  Missing values: 0")
    print(f"  Infinite vals : 0")
    print(f"  Classes       : {sorted(labels)}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("AI PHISHING DETECTION")
    print("TRAINING DATASET GENERATOR")
    print("=" * 60)

    # Create directories
    create_directories()

    # --------------------------------------------------------
    # Load original PhiUSIIL dataset
    # --------------------------------------------------------

    phiusill_df = load_phiusill_dataset()

    # --------------------------------------------------------
    # Create raw_url.csv
    # --------------------------------------------------------

    raw_df = create_raw_url_dataset(
        phiusill_df
    )

    # --------------------------------------------------------
    # Extract 22 features
    # --------------------------------------------------------

    records = extract_features_from_raw_dataset(
        raw_df
    )

    # --------------------------------------------------------
    # Create final dataset
    # --------------------------------------------------------

    final_df = create_final_dataset(
        records
    )

    # --------------------------------------------------------
    # Validate final dataset
    # --------------------------------------------------------

    validate_final_dataset(
        final_df
    )

    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETED")
    print("=" * 60)

    print("\nCreated files:")

    print(f"1. Raw URL dataset:")
    print(f"   {RAW_URL_DATASET}")

    print(f"\n2. Final feature dataset:")
    print(f"   {FINAL_DATASET}")

    print("\nPipeline completed successfully! 🚀")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
