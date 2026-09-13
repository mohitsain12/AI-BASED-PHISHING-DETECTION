"""
train_model.py

Train and evaluate the Random Forest phishing detection model.

INPUT:
    data/processed/X_train.csv
    data/processed/X_test.csv
    data/processed/y_train.csv
    data/processed/y_test.csv

MODEL:
    Random Forest Classifier
    Defined in model.py

OUTPUT:
    models/phishing_model.pkl
    models/feature_importance.png
    models/model_metrics.png
    models/confusion_matrix.png

LABEL:
    0 = Phishing
    1 = Legitimate
"""


import os
import time
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from model import (
    create_model,
    get_model_parameters,
    print_model_configuration
)

from feature_extractor import ALL_FEATURES


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

MODEL_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "models"
)

MODEL_FILE = (
    os.path.join(
        MODEL_DIRECTORY,
        "phishing_model.pkl"
    )
)

FEATURE_IMPORTANCE_PLOT = (
    os.path.join(
        MODEL_DIRECTORY,
        "feature_importance.png"
    )
)

METRICS_PLOT = (
    os.path.join(
        MODEL_DIRECTORY,
        "model_metrics.png"
    )
)

CONFUSION_MATRIX_PLOT = (
    os.path.join(
        MODEL_DIRECTORY,
        "confusion_matrix.png"
    )
)


# ============================================================
# EXPECTED FILES
# ============================================================

X_TRAIN_FILE = (
    os.path.join(
        DATA_DIRECTORY,
        "X_train.csv"
    )
)

X_TEST_FILE = (
    os.path.join(
        DATA_DIRECTORY,
        "X_test.csv"
    )
)

Y_TRAIN_FILE = (
    os.path.join(
        DATA_DIRECTORY,
        "y_train.csv"
    )
)

Y_TEST_FILE = (
    os.path.join(
        DATA_DIRECTORY,
        "y_test.csv"
    )
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIRECTORY,
    exist_ok=True
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data():
    """
    Load preprocessed training and testing datasets.
    """

    print("\n" + "=" * 70)
    print("LOADING PREPROCESSED DATA")
    print("=" * 70)

    required_files = [
        X_TRAIN_FILE,
        X_TEST_FILE,
        Y_TRAIN_FILE,
        Y_TEST_FILE
    ]

    for file_path in required_files:

        if not os.path.exists(
            file_path
        ):

            raise FileNotFoundError(
                f"\nRequired file not found:\n"
                f"{file_path}\n\n"
                f"Run data_preprocessing.py first."
            )

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------

    X_train = pd.read_csv(
        X_TRAIN_FILE
    )

    X_test = pd.read_csv(
        X_TEST_FILE
    )

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    y_train = pd.read_csv(
        Y_TRAIN_FILE
    ).squeeze("columns")

    y_test = pd.read_csv(
        Y_TEST_FILE
    ).squeeze("columns")

    print(
        f"\nX_train shape : {X_train.shape}"
    )

    print(
        f"X_test shape  : {X_test.shape}"
    )

    print(
        f"y_train shape : {y_train.shape}"
    )

    print(
        f"y_test shape  : {y_test.shape}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# VALIDATE TRAINING DATA
# ============================================================

def validate_training_data(
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Validate training/testing data before fitting the model.
    """

    print("\n" + "=" * 70)
    print("VALIDATING TRAINING DATA")
    print("=" * 70)

    # ========================================================
    # FEATURE COUNT
    # ========================================================

    if X_train.shape[1] != 22:

        raise ValueError(
            f"X_train must contain 22 features. "
            f"Found {X_train.shape[1]}."
        )

    if X_test.shape[1] != 22:

        raise ValueError(
            f"X_test must contain 22 features. "
            f"Found {X_test.shape[1]}."
        )

    print(
        "✓ Training data contains 22 features."
    )

    # ========================================================
    # FEATURE ORDER
    # ========================================================

    if list(X_train.columns) != ALL_FEATURES:

        raise ValueError(
            "\nX_train feature order does not match "
            "feature_extractor.py.\n\n"
            f"Expected:\n{ALL_FEATURES}\n\n"
            f"Found:\n{list(X_train.columns)}"
        )

    if list(X_test.columns) != ALL_FEATURES:

        raise ValueError(
            "X_test feature order does not match "
            "feature_extractor.py."
        )

    print(
        "✓ Feature order is correct."
    )

    # ========================================================
    # TRAIN / TEST FEATURE CONSISTENCY
    # ========================================================

    if list(X_train.columns) != list(
        X_test.columns
    ):

        raise ValueError(
            "Training and testing feature columns differ."
        )

    print(
        "✓ Train/test feature structure matches."
    )

    # ========================================================
    # SAMPLE COUNTS
    # ========================================================

    if len(X_train) != len(y_train):

        raise ValueError(
            "X_train and y_train contain different "
            "numbers of samples."
        )

    if len(X_test) != len(y_test):

        raise ValueError(
            "X_test and y_test contain different "
            "numbers of samples."
        )

    print(
        "✓ Training sample counts match."
    )

    print(
        "✓ Testing sample counts match."
    )

    # ========================================================
    # NUMERIC VALUES
    # ========================================================

    for feature in ALL_FEATURES:

        if not pd.api.types.is_numeric_dtype(
            X_train[feature]
        ):

            raise ValueError(
                f"Training feature '{feature}' "
                f"is not numeric."
            )

        if not pd.api.types.is_numeric_dtype(
            X_test[feature]
        ):

            raise ValueError(
                f"Testing feature '{feature}' "
                f"is not numeric."
            )

    print(
        "✓ All features are numeric."
    )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    if X_train.isna().any().any():

        raise ValueError(
            "NaN values found in X_train."
        )

    if X_test.isna().any().any():

        raise ValueError(
            "NaN values found in X_test."
        )

    if y_train.isna().any():

        raise ValueError(
            "NaN values found in y_train."
        )

    if y_test.isna().any():

        raise ValueError(
            "NaN values found in y_test."
        )

    print(
        "✓ No missing values."
    )

    # ========================================================
    # INFINITE VALUES
    # ========================================================

    if np.isinf(
        X_train.to_numpy()
    ).any():

        raise ValueError(
            "Infinite values found in X_train."
        )

    if np.isinf(
        X_test.to_numpy()
    ).any():

        raise ValueError(
            "Infinite values found in X_test."
        )

    print(
        "✓ No infinite values."
    )

    # ========================================================
    # LABEL VALIDATION
    # ========================================================

    train_labels = set(
        y_train.astype(int).unique()
    )

    test_labels = set(
        y_test.astype(int).unique()
    )

    if train_labels != {0, 1}:

        raise ValueError(
            f"Invalid training labels: "
            f"{train_labels}"
        )

    if test_labels != {0, 1}:

        raise ValueError(
            f"Invalid testing labels: "
            f"{test_labels}"
        )

    print(
        "✓ Training labels: 0 and 1."
    )

    print(
        "✓ Testing labels: 0 and 1."
    )

    # ========================================================
    # DISPLAY DISTRIBUTION
    # ========================================================

    print("\nTraining distribution:")

    print(
        y_train
        .value_counts()
        .sort_index()
    )

    print("\nTesting distribution:")

    print(
        y_test
        .value_counts()
        .sort_index()
    )

    print(
        "\n✓ TRAINING DATA VALIDATION PASSED."
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train
):
    """
    Create and train the Random Forest model.
    """

    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    # --------------------------------------------------------
    # Display configuration
    # --------------------------------------------------------

    print_model_configuration()

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model()

    print(
        "\nStarting training..."
    )

    start_time = time.time()

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    end_time = time.time()

    training_time = (
        end_time
        - start_time
    )

    print(
        "\n✓ Random Forest training completed."
    )

    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )

    return model, training_time


# ============================================================
# MAKE PREDICTIONS
# ============================================================

def make_predictions(
    model,
    X_test
):
    """
    Generate predictions and probabilities.
    """

    print("\n" + "=" * 70)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 70)

    # --------------------------------------------------------
    # Class predictions
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Prediction probabilities
    # --------------------------------------------------------

    y_probability = (
        model.predict_proba(
            X_test
        )
    )

    print(
        f"Predictions generated: "
        f"{len(y_pred):,}"
    )

    return (
        y_pred,
        y_probability
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    y_test,
    y_pred,
    training_time
):
    """
    Calculate classification metrics.
    """

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # Precision
    #
    # Positive class = 1 (Legitimate)
    # --------------------------------------------------------

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    # --------------------------------------------------------
    # F1 score
    # --------------------------------------------------------

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print(
        f"\nAccuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-Score  : {f1:.4f}"
    )

    print(
        f"Training Time : "
        f"{training_time:.2f} seconds"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT")
    print("-" * 70)

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Phishing",
            "Legitimate"
        ],
        zero_division=0
    )

    print(
        report
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    print(
        "\n" + "-" * 70
    )

    print(
        "CONFUSION MATRIX"
    )

    print(
        "-" * 70
    )

    print(
        "\n                    Predicted"
    )

    print(
        "                  Phishing  Legitimate"
    )

    print(
        f"Actual Phishing    "
        f"{matrix[0][0]:8d}"
        f"{matrix[0][1]:11d}"
    )

    print(
        f"Actual Legitimate  "
        f"{matrix[1][0]:8d}"
        f"{matrix[1][1]:11d}"
    )

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Training_Time_Seconds": training_time
    }

    return (
        metrics,
        matrix
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def calculate_feature_importance(
    model
):
    """
    Calculate Random Forest feature importance.
    """

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    importance = model.feature_importances_

    feature_importance = pd.DataFrame(
        {
            "Feature": ALL_FEATURES,
            "Importance": importance
        }
    )

    feature_importance = (
        feature_importance
        .sort_values(
            by="Importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\nFeature importance ranking:\n"
    )

    for index, row in (
        feature_importance
        .iterrows()
    ):

        print(
            f"{index + 1:02d}. "
            f"{row['Feature']:<28} "
            f"{row['Importance']:.6f}"
        )

    return feature_importance


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):
    """
    Save trained Random Forest model using pickle.
    """

    print("\n" + "=" * 70)
    print("SAVING TRAINED MODEL")
    print("=" * 70)

    with open(
        MODEL_FILE,
        "wb"
    ) as file:

        pickle.dump(
            model,
            file
        )

    print(
        f"\n✓ Model saved:"
    )

    print(
        f"  {MODEL_FILE}"
    )


# ============================================================
# PLOT FEATURE IMPORTANCE
# ============================================================

def plot_feature_importance(
    feature_importance
):
    """
    Plot and save feature importance as a PNG image.
    """

    plot_data = feature_importance.sort_values(
        by="Importance",
        ascending=True
    )

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    axis.barh(
        plot_data["Feature"],
        plot_data["Importance"],
        color="steelblue"
    )

    axis.set_title("Random Forest Feature Importance")
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    figure.tight_layout()
    figure.savefig(FEATURE_IMPORTANCE_PLOT, dpi=150)
    plt.close(figure)

    print("✓ Feature importance plot saved:")
    print(f"  {FEATURE_IMPORTANCE_PLOT}")


# ============================================================
# PLOT METRICS
# ============================================================

def plot_metrics(metrics):
    """
    Plot and save evaluation metrics as a PNG image.
    """

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score"
    ]

    metric_values = [
        metrics[name]
        for name in metric_names
    ]

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    bars = axis.bar(
        metric_names,
        metric_values,
        color=[
            "#4472c4",
            "#70ad47",
            "#ed7d31",
            "#a64d79"
        ]
    )

    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Random Forest Evaluation Metrics")
    axis.bar_label(bars, fmt="%.3f", padding=3)
    figure.tight_layout()
    figure.savefig(METRICS_PLOT, dpi=150)
    plt.close(figure)

    print("✓ Metrics plot saved:")
    print(f"  {METRICS_PLOT}")


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    matrix
):
    """
    Plot and save the confusion matrix as a PNG image.
    """

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    image = axis.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues"
    )

    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Phishing", "Legitimate"],
        yticklabels=["Phishing", "Legitimate"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title="Confusion Matrix"
    )

    threshold = matrix.max() / 2

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:d}",
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] > threshold else "black"
            )

    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PLOT, dpi=150)
    plt.close(figure)

    print("✓ Confusion matrix plot saved:")
    print(f"  {CONFUSION_MATRIX_PLOT}")


# ============================================================
# VERIFY SAVED MODEL
# ============================================================

def verify_saved_model():
    """
    Load the saved model again to make sure the file
    is valid and usable.
    """

    print("\n" + "=" * 70)
    print("VERIFYING SAVED MODEL")
    print("=" * 70)

    if not os.path.exists(
        MODEL_FILE
    ):

        raise FileNotFoundError(
            "Saved model file was not created."
        )

    with open(
        MODEL_FILE,
        "rb"
    ) as file:

        loaded_model = pickle.load(
            file
        )

    # --------------------------------------------------------
    # Verify Random Forest
    # --------------------------------------------------------

    if not hasattr(
        loaded_model,
        "predict"
    ):

        raise ValueError(
            "Saved object is not a valid ML model."
        )

    if not hasattr(
        loaded_model,
        "predict_proba"
    ):

        raise ValueError(
            "Saved model does not support probability prediction."
        )

    print(
        "✓ Model file exists."
    )

    print(
        "✓ Model can be loaded."
    )

    print(
        "✓ Model supports prediction."
    )

    print(
        "✓ Model supports probability prediction."
    )

    return loaded_model


# ============================================================
# FINAL SUMMARY
# ============================================================

def display_final_summary(
    model,
    metrics,
    feature_importance
):
    """
    Display final training summary.
    """

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "\nMODEL"
    )

    print(
        "Random Forest Classifier"
    )

    print(
        f"Number of trees: "
        f"{model.n_estimators}"
    )

    print(
        "\nPERFORMANCE"
    )

    print(
        f"Accuracy  : "
        f"{metrics['Accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{metrics['Precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{metrics['Recall']:.4f}"
    )

    print(
        f"F1-Score  : "
        f"{metrics['F1_Score']:.4f}"
    )

    print(
        "\nTOP 5 FEATURES"
    )

    for index, row in (
        feature_importance
        .head(5)
        .iterrows()
    ):

        print(
            f"{index + 1}. "
            f"{row['Feature']} "
            f"({row['Importance']:.4f})"
        )

    print(
        "\nMODEL FILE"
    )

    print(
        MODEL_FILE
    )

    print(
        "\n✓ Random Forest model is ready for prediction."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("AI PHISHING DETECTION")
    print("RANDOM FOREST TRAINING PIPELINE")
    print("=" * 70)

    # ========================================================
    # STEP 1
    # Load data
    # ========================================================

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_training_data()

    # ========================================================
    # STEP 2
    # Validate data
    # ========================================================

    validate_training_data(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ========================================================
    # STEP 3
    # Train model
    # ========================================================

    (
        model,
        training_time
    ) = train_model(
        X_train,
        y_train
    )

    # ========================================================
    # STEP 4
    # Make predictions
    # ========================================================

    (
        y_pred,
        y_probability
    ) = make_predictions(
        model,
        X_test
    )

    # ========================================================
    # STEP 5
    # Evaluate model
    # ========================================================

    (
        metrics,
        confusion
    ) = evaluate_model(
        y_test,
        y_pred,
        training_time
    )

    # ========================================================
    # STEP 6
    # Feature importance
    # ========================================================

    feature_importance = (
        calculate_feature_importance(
            model
        )
    )

    # ========================================================
    # STEP 7
    # Save model
    # ========================================================

    save_model(
        model
    )

    # ========================================================
    # STEP 8
    # Plot feature importance
    # ========================================================

    plot_feature_importance(
        feature_importance
    )

    # ========================================================
    # STEP 9
    # Plot metrics
    # ========================================================

    plot_metrics(
        metrics
    )

    # ========================================================
    # STEP 10
    # Plot confusion matrix
    # ========================================================

    plot_confusion_matrix(
        confusion
    )

    # ========================================================
    # STEP 11
    # Verify model
    # ========================================================

    verify_saved_model()

    # ========================================================
    # STEP 12
    # Final summary
    # ========================================================

    display_final_summary(
        model,
        metrics,
        feature_importance
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()