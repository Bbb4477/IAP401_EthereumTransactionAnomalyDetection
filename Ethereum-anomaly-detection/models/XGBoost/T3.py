import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier  # For type hinting


# Load and clean REAL Etherscan data (single row)
def load_real_data(csv_path):
    """
    Load single-row Etherscan data and clean it to match training pipeline
    """
    df = pd.read_csv(csv_path)
    print(f"Raw data loaded with shape: {df.shape}")

    # Clean column names (matching notebook)
    df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True)
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    print(f"Cleaned column names: {len(df.columns)} columns")

    # Drop metadata and problematic categorical columns (same as training)
    columns_to_drop = [
        'Unnamed:_0', 'Index', 'Address',  # Metadata
        'ERC20_most_sent_token_type',  # Categorical 1
        'ERC20_most_rec_token_type'  # Categorical 2 (note: yours has this, not _type)
    ]
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    print(f"Dropping columns: {columns_to_drop}")

    df_clean = df.drop(columns=columns_to_drop, errors='ignore')

    # Fill NaNs with 0 (same as training)
    df_clean = df_clean.fillna(0)
    print(df_clean.info())
    # Ensure all numeric (force float64)
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)

    print(f"Final cleaned data shape: {df_clean.shape}")
    print(f"All numeric: {df_clean.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()}")

    return df_clean


# Main script
if __name__ == "__main__":
    csv_path = "input.csv"  # Your Etherscan output file
    model_path = "fraud_model.pkl"  # Your trained model

    # Check if files exist
    try:
        with open(model_path, 'rb') as f:
            pass  # Just checking if model exists
        print(f"✓ Model file found: {model_path}")
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        print("Run the pickle dump in your notebook first!")
        exit(1)

    # Load real Etherscan data
    try:
        df_real = load_real_data(csv_path)
    except FileNotFoundError:
        print(f"❌ Input file not found: {csv_path}")
        print("Please ensure input.csv exists in the same directory")
        exit(1)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        exit(1)

    # Since it's a single row, extract it
    if len(df_real) != 1:
        print(f"⚠️  Warning: Expected 1 row, got {len(df_real)} rows")

    row_data = df_real.iloc[0]
    address = df_real.iloc[0].get('Address', 'Unknown')  # Try to get address if not dropped

    print(f"\n{'=' * 50}")
    print(f"PREDICTING FOR ETHERSCAN DATA")
    print(f"{'=' * 50}")
    print(f"Address: {address}")
    print(f"Contract: {'Yes' if row_data.get('Contract', 0) == 1 else 'No'}")
    print(f"Total Transactions: {row_data.get('total_transactions_(including_tnx_to_create_contract', 0)}")
    print(f"Total Ether Sent: {row_data.get('total_Ether_sent', 0):.6f}")
    print(f"ERC20 Transactions: {row_data.get('Total_ERC20_tnxs', 0)}")
    print(f"{'=' * 50}")

    # Load the trained model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit(1)

    # Prepare features (drop FLAG if it existed, but it won't for real data)
    features = row_data.drop(labels=['FLAG'], errors='ignore')

    # Ensure we have the right number of features (46 for training)
    if len(features) != 46:
        print(f"⚠️  Warning: Expected 46 features, got {len(features)}")
        print("Available features:", list(features.index))

    # Convert to float and reshape for prediction
    features = features.astype(float).values.reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    # Format output
    fraud_prob = prob[1] * 100
    non_fraud_prob = prob[0] * 100

    print(f"\n🎯 PREDICTION RESULTS")
    print(f"{'-' * 30}")
    print(f"Prediction: {'🔔 FRAUD (1)' if prediction == 1 else '✅ NON-FRAUD (0)'}")
    print(f"Confidence: {max(fraud_prob, non_fraud_prob):.1f}%")
    print(f"Fraud Probability: {fraud_prob:.2f}%")
    print(f"Non-Fraud Probability: {non_fraud_prob:.2f}%")

    if prediction == 1:
        print(f"\n🚨 ALERT: This address shows fraudulent patterns!")
        print(f"   • High fraud probability: {fraud_prob:.1f}%")
        if fraud_prob > 95:
            print(f"   • CRITICAL: Extremely high confidence ({fraud_prob:.1f}%)")
        elif fraud_prob > 80:
            print(f"   • HIGH: Strong fraud indicators ({fraud_prob:.1f}%)")
    else:
        print(f"\n✅ SAFE: This address appears legitimate")
        print(f"   • Low fraud probability: {fraud_prob:.1f}%")
        if fraud_prob > 10:
            print(f"   • MONITOR: Some suspicious activity detected ({fraud_prob:.1f}%)")

    print(f"\n📊 Raw Probabilities: Non-Fraud={prob[0]:.4f}, Fraud={prob[1]:.4f}")
    print(f"{'=' * 50}")