"""
model.py

Random Forest model definition for the
AI Phishing Detection Project.

This file contains ONLY the machine-learning
model configuration.

Training and evaluation are handled separately
by train_model.py.

MODEL:
    Random Forest Classifier

INPUT:
    22 numerical phishing-detection features

OUTPUT:
    0 = Phishing
    1 = Legitimate
"""


from sklearn.ensemble import RandomForestClassifier


# ============================================================
# MODEL CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 300

MAX_DEPTH = None

MIN_SAMPLES_SPLIT = 2

MIN_SAMPLES_LEAF = 1

MAX_FEATURES = "sqrt"

CLASS_WEIGHT = None

N_JOBS = -1


# ============================================================
# CREATE RANDOM FOREST MODEL
# ============================================================

def create_model():
    """
    Create and return the Random Forest classifier.

    Returns
    -------
    RandomForestClassifier
        Configured Random Forest model.
    """

    model = RandomForestClassifier(

        # ----------------------------------------------------
        # Number of decision trees
        # ----------------------------------------------------

        n_estimators=N_ESTIMATORS,

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        random_state=RANDOM_STATE,

        # ----------------------------------------------------
        # Maximum depth of each tree
        #
        # None means trees expand until stopping criteria
        # are reached.
        # ----------------------------------------------------

        max_depth=MAX_DEPTH,

        # ----------------------------------------------------
        # Minimum samples required to split a node
        # ----------------------------------------------------

        min_samples_split=MIN_SAMPLES_SPLIT,

        # ----------------------------------------------------
        # Minimum samples required at a leaf
        # ----------------------------------------------------

        min_samples_leaf=MIN_SAMPLES_LEAF,

        # ----------------------------------------------------
        # Number of features considered at each split
        #
        # sqrt is a standard Random Forest configuration
        # for classification.
        # ----------------------------------------------------

        max_features=MAX_FEATURES,

        # ----------------------------------------------------
        # Class weighting
        #
        # None is appropriate because our generated dataset
        # is balanced.
        # ----------------------------------------------------

        class_weight=CLASS_WEIGHT,

        # ----------------------------------------------------
        # Use all available CPU cores
        # ----------------------------------------------------

        n_jobs=N_JOBS
    )

    return model


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_parameters():
    """
    Return model configuration as a dictionary.

    Useful for logging and experiment tracking.
    """

    return {
        "model": "Random Forest Classifier",
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "max_features": MAX_FEATURES,
        "class_weight": CLASS_WEIGHT,
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS
    }


# ============================================================
# DISPLAY MODEL CONFIGURATION
# ============================================================

def print_model_configuration():
    """
    Display the Random Forest configuration.
    """

    print("\n" + "=" * 60)
    print("RANDOM FOREST MODEL CONFIGURATION")
    print("=" * 60)

    parameters = get_model_parameters()

    for parameter, value in parameters.items():

        print(
            f"{parameter:<22}: {value}"
        )

    print("=" * 60)


# ============================================================
# TEST MODEL CREATION
# ============================================================

if __name__ == "__main__":

    print_model_configuration()

    model = create_model()

    print(
        "\n✓ Random Forest model created successfully."
    )

    print(
        f"✓ Number of trees: {model.n_estimators}"
    )

    print(
        f"✓ Random state: {model.random_state}"
    )

    print(
        "\nModel is ready for train_model.py."
    )
