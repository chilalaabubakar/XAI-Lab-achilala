import os
import pandas as pd

# Paths
# Note: If your file is still inside the data folder, change this to "data/credit_dataset1.csv"
DATA_PATH = "data/credit_dataset2.csv"  
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
VIZ_DIR = "visualization"

# Dataset Configurations
TARGET = "Risk"
PROTECTED_ATTR = "Sex"
WITH_GENDER = True
RANDOM_STATE = 42

# Feature Definitions mapped EXACTLY to your dataset headers
CATEGORICAL_COLS = [
    "Sex", 
    "Housing", 
    "Saving accounts", 
    "Checking account", 
    "Purpose"
]

NUMERICAL_COLS = [
    "Age", 
    "Job", 
    "Credit amount", 
    "Duration"
]

def load_data():
    """Loads the dataset and ensures columns match the required schema."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please ensure it is in the right directory.")
    df = pd.read_csv(DATA_PATH)
    return df

def encode_target(y):
    """Encodes the Good/Bad string labels into 1/0 numeric binary format."""
    return y.map({"Good": 1, "Bad": 0}).astype(int)
