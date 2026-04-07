import os
import pandas as pd

# Paths
DATA_PATH = "credit_dataset1.csv"  # Ensure this file is in your project root
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
VIZ_DIR = "visualization"

# Dataset Configurations
TARGET = "Label"
PROTECTED_ATTR = "Sex"
WITH_GENDER = True
RANDOM_STATE = 42

# Feature Definitions based on the German Credit Data
CATEGORICAL_COLS = [
    "Sex", 
    "Housing", 
    "Saving_accounts", 
    "Checking_account", 
    "Purpose"
]

NUMERICAL_COLS = [
    "Age", 
    "Job", 
    "Credit_amount", 
    "Duration"
]

def load_data():
    """Loads the dataset and ensures columns match the required schema."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please ensure it is in the root directory.")
    df = pd.read_csv(DATA_PATH)
    return df

def encode_target(y):
    """Encodes the Good/Bad string labels into 1/0 numeric binary format."""
    return y.map({"Good": 1, "Bad": 0}).astype(int)
