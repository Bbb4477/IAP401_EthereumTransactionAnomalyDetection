import pandas as pd
import pickle
import shap
import numpy as np


# Function to process the input CSV and perform predictions with SHAP explanations
def process_and_predict(input_csv_path, model_path='XGBoost.pkl'):
    # Load the XGBoost model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("[LOG]Model Loaded")
    # Create SHAP explainer from the loaded model
    explainer = shap.TreeExplainer(model)

    # Load the input CSV
    df = pd.read_csv(input_csv_path)

    # Clean column names (replicate regex strip and replace spaces with underscores)
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]

    # Check if FLAG is present for true labels
    has_flag = 'FLAG' in df.columns
    if has_flag:
        y = df['FLAG']
        X = df.drop(['FLAG'], axis=1)
    else:
        y = None
        X = df.copy()

    # Drop unnecessary columns if present
    cols_to_drop = ['Unnamed:_0', 'Index', 'Address']
    X = X.drop([col for col in cols_to_drop if col in X.columns], axis=1)
    print(X.info())

    # Handle NaN in categorical columns (fill with space as inferred from feature names)
    cat_cols = ['ERC20_most_sent_token_type', 'ERC20_most_rec_token_type']
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].fillna(' ')

    # One-hot encode categorical columns
    X_encoded = pd.get_dummies(X, columns=[col for col in cat_cols if col in X.columns])

    # Handle NaN in numerical columns (fill with 0 as common for ERC20 features)
    X_encoded = X_encoded.fillna(0)
    # Loop through each row and predict (only misclassified if FLAG present, else all)
    for i in range(0,len(X_encoded)):
        example_row = X_encoded.iloc[[i]]
        pred_label = model.predict(example_row)[0]

        if has_flag:
            true_label = y.iloc[i]
            if true_label == pred_label:
                continue  # Skip correctly classified
        else:
            true_label = 'Unknown'

        # Get SHAP values
        shap_values = explainer(example_row)
        contrib = shap_values.values[0]  # Contributions towards fraud (positive for class 1)

        # Create DataFrame for feature contributions
        fraud_features = pd.DataFrame({
            "feature": X_encoded.columns,
            "contribution": contrib
        }).sort_values(by="contribution", ascending=False)

        # Prediction probabilities
        prob = model.predict_proba(example_row)[0]

        # Print the output structure
        print(f"Index: {i}")
        print(f"Prediction: {'Fraud (1)' if pred_label == 1 else 'Non-Fraud (0)'}")
        print(f"Probability (Non-Fraud, Fraud): [{prob[0]:.4f}, {prob[1]:.4f}]")
        print(f"True label (FLAG): {true_label} (1=fraud, 0=non-fraud)")
        print("Top fraud-driving features:")
        top_fraud = fraud_features.sort_values(by="contribution", ascending=False).head(5)
        top_nonfraud = fraud_features.sort_values(by="contribution", ascending=True).head(5)
        print(top_fraud)
        print("       -------------------------------")
        print(top_nonfraud)
        print("--------------------------------------------------")
        print("\n")

# Example usage (adjust paths as needed)
# process_and_predict('input.csv', 'XGBoost.pkl')

def main():
    process_and_predict("input.csv",)

if __name__ == "__main__":
    main()