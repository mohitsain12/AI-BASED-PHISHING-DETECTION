# AI Phishing Detection

A Random Forest phishing URL detector based on the PhiUSIIL phishing URL dataset.

## Project Pipeline

1. Load and clean the original dataset.
2. Select a balanced sample of 5,000 URLs:
   - 2,500 phishing URLs (`label = 0`)
   - 2,500 legitimate URLs (`label = 1`)
3. Extract 22 URL and online HTML/JavaScript features.
4. Split the data into training and testing sets.
5. Train and evaluate a Random Forest classifier.
6. Predict new URLs interactively.

## Requirements

Use Python 3. Install the required packages:

```powershell
pip install pandas numpy requests scikit-learn matplotlib
```

## Dataset

Place the original dataset at:

```text
data/PhiUSIIL_Phishing_URL_Dataset.csv
```

The source dataset must contain URL and label columns. Labels use this convention:

- `0` = phishing
- `1` = legitimate

## Run the Pipeline

Run commands from the project root (`D:\project_ai`):

### 1. Generate Features

```powershell
python .\src\generate_training_dataset.py
```

This creates:

- `data/raw_url.csv`
- `data/phishing_22_features.csv`
- `data/phishing_22_features_checkpoint.csv`

The checkpoint file is a progress backup. The final dataset is `phishing_22_features.csv`.

The HTML/JavaScript features are extracted by making online requests to the URLs. If a page cannot be downloaded, those HTML/JavaScript features are recorded as zero.

### 2. Preprocess Data

```powershell
python .\src\data_preprocessing.py
```

This creates:

- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`

The split uses 80% training data and 20% testing data with stratification.

### 3. Train the Model

```powershell
python .\src\train_model.py
```

This creates:

- `models/phishing_model.pkl`
- `models/feature_importance.png`
- `models/model_metrics.png`
- `models/confusion_matrix.png`

The Random Forest uses 300 trees and all available CPU cores.

### 4. Predict a URL

```powershell
python .\src\predict.py
```

Enter a URL when prompted. Type `exit` to stop.

Prediction results include:

- Phishing or legitimate classification
- Phishing probability
- Legitimate probability
- All 22 extracted features

## Features

The model uses 22 numerical features covering:

- URL length and structure
- Domain and top-level domain properties
- HTTPS and IP address usage
- Obfuscation and special characters
- Iframes, popups, hidden fields, redirects, and forms
- Password fields and social-network links

## Important Notes

- Feature extraction performs live HTTP requests and may take time.
- A short request timeout is used so unavailable websites do not block the pipeline indefinitely.
- Random Forest does not require feature scaling.
- The generated dataset and model outputs are local artifacts and should be regenerated when the source data or feature logic changes.
