import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Load data from a CSV file.
    """
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """
    Preprocess the data:
    - Map Extracurricular Activities to 0 and 1.
    - Split into features (X) and target (y).
    - Return train/test split.
    """
    # Create a copy to avoid SettingWithCopyWarning if slice
    df = df.copy()
    
    # Map categorical data as seen in the notebook
    if 'Extracurricular Activities' in df.columns:
        df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    
    # Define features and target
    X = df.drop("Performance Index", axis=1)
    y = df["Performance Index"]

    # Return split data
    return train_test_split(X, y, test_size=0.2, random_state=42)
