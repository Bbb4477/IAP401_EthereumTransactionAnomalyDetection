import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier  # For type hinting, but not strictly needed


# Load and clean the dataset (matching notebook logic)
def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)

    # Clean column names
    df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True)
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]

    # Drop metadata and categorical columns
    columns_to_drop = [
        'Unnamed:_0', 'Index', 'Address',
        'ERC20_most_sent_token_type',
        'ERC20_most_rec_token_type'
    ]
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df_clean = df.drop(columns=columns_to_drop, errors='ignore')

    # Fill NaNs with 0
    df_clean = df_clean.fillna(0)

    return df_clean


# Main script
if __name__ == "__main__":
    csv_path = "transaction_dataset_V1.csv"  # Adjust if needed
    model_path = "fraud_model.pkl"  # Path to your pickled model

    # Load data
    df_clean = load_and_clean_data(csv_path)
    print(f"Loaded dataset with shape: {df_clean.shape}")

    # Load the pickled model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")

    # Get index from user input
    try:
        index = int(input("Enter the row index to predict (0 to 9840): "))
        if index < 0 or index >= len(df_clean):
            raise ValueError("Index out of range")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    # Prepare features for the row (drop 'FLAG' if present)
    row = df_clean.iloc[index]
    features = row.drop(labels=['FLAG'], errors='ignore')  # Ignore if 'FLAG' not present
    features = features.astype(float).values.reshape(1, -1)

    # Predict
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    # Output
    print(f"\nIndex: {index}")
    print(f"Prediction: {'Fraud (1)' if prediction == 1 else 'Non-Fraud (0)'}")
    print(f"Probability (Non-Fraud, Fraud): [{prob[0]:.4f}, {prob[1]:.4f}]")
    if 'FLAG' in row:
        print(f"True label (FLAG): {row['FLAG']} (for verification)")