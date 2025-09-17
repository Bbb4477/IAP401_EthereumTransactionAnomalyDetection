import pandas as pd
import pickle
import shap
import numpy as np


def process_and_predict(input_csv_path, model_path='XGBoost.pkl'):
    # Load the XGBoost model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Get the feature names the model expects
    model_feature_names = model.get_booster().feature_names
    print("Model's expected feature names:")
    print(model_feature_names)
    print(f"Number of expected features: {len(model_feature_names)}")
    print("--------------------------------------------------")

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Load the input CSV
    df = pd.read_csv(input_csv_path)

    # Clean column names
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]

    # Check if FLAG is present
    has_flag = 'FLAG' in df.columns
    if has_flag:
        y = df['FLAG']
        X = df.drop(['FLAG'], axis=1)
    else:
        y = None
        X = df.copy()

    # Drop unnecessary columns
    cols_to_drop = ['Unnamed:_0', 'Index', 'Address']
    X = X.drop([col for col in cols_to_drop if col in X.columns], axis=1)

    # Handle NaN in categorical columns
    cat_cols = ['ERC20_most_sent_token_type', 'ERC20_most_rec_token_type']
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].fillna(' ')

    # One-hot encode categorical columns
    X_encoded = pd.get_dummies(X, columns=[col for col in cat_cols if col in X.columns])

    # Handle NaN in numerical columns
    X_encoded = X_encoded.fillna(0)

    # Print input data feature names after encoding
    print("Input data feature names after encoding:")
    print(list(X_encoded.columns))
    print(f"Number of input features: {len(X_encoded.columns)}")
    print("--------------------------------------------------")

    # Check for feature mismatch
    input_features = set(X_encoded.columns)
    model_features = set(model_feature_names)
    missing_features = model_features - input_features
    extra_features = input_features - model_features

    if missing_features or extra_features:
        print("Feature mismatch detected!")
        if missing_features:
            print("Features in model but missing in input data:")
            print(missing_features)
        if extra_features:
            print("Features in input data but not in model:")
            print(extra_features)
        raise ValueError("Feature names mismatch between model and input data.")

    # Loop through each row and predict
    for i in range(len(X_encoded)):
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
        contrib = shap_values.values[0]  # Contributions towards fraud

        # Create DataFrame for feature contributions
        fraud_features = pd.DataFrame({
            "feature": X_encoded.columns,
            "contribution": contrib
        }).sort_values(by="contribution", ascending=False)

        # Prediction probabilities
        prob = model.predict_proba(example_row)[0]

        # Print output
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


def main():
    try:
        process_and_predict('input.csv', 'XGBoost.pkl')
    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()