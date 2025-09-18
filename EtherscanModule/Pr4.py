import pandas as pd
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

def load_pickle_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)

try:
    model = load_pickle_file('XGBoost.pkl')
    encoder = load_pickle_file('encoder.pkl')
    feature_names = load_pickle_file('feature_names.pkl')
except FileNotFoundError as e:
    print(e)
    print("Ensure 'XGBoost.pkl', 'encoder.pkl', and 'feature_names.pkl' are in the current directory.")
    exit(1)
except Exception as e:
    print(f"Error loading files: {e}")
    exit(1)

# Print expected columns
print(f"Expected features ({len(feature_names)}):")
for i, col in enumerate(feature_names, 1):
    print(f"  {i}. {col}")

try:
    new_df = pd.read_csv('input.csv')
except FileNotFoundError:
    print("Input file 'input.csv' not found.")
    exit(1)
except Exception as e:
    print(f"Error reading input.csv: {e}")
    exit(1)

try:
    print("\nInput CSV columns:", list(new_df.columns))
    new_df.columns = [col.strip().replace(' ', '_') for col in new_df.columns]
    print("Columns after standardization:", list(new_df.columns))

    columns_to_drop = ['Unnamed:_0', 'Index', 'Address']
    new_df.drop([col for col in columns_to_drop if col in new_df.columns], axis=1, inplace=True)

    if 'FLAG' in new_df.columns:
        new_df.drop('FLAG', axis=1, inplace=True)

    cat_features = ['ERC20_most_sent_token_type', 'ERC20_most_rec_token_type']
    num_features = [col for col in new_df.columns if col not in cat_features]

    print(f"Numerical features ({len(num_features)}):", num_features)
    print(f"Categorical features ({len(cat_features)}):", cat_features)

    # Fill categorical columns with 'UNKNOWN' for consistency
    new_df[cat_features] = new_df[cat_features].fillna('UNKNOWN').replace('', 'UNKNOWN').replace('None', 'UNKNOWN')
    new_df[num_features] = new_df[num_features].fillna(0)

    new_df_encoded = encoder.transform(new_df)
    print("Columns after encoding:", list(new_df_encoded.columns))
    print("Encoded shape:", new_df_encoded.shape)

    new_df_encoded = new_df_encoded.fillna(0)
    new_df_encoded = new_df_encoded.reindex(columns=feature_names, fill_value=0)

    print("Final encoded shape:", new_df_encoded.shape)
    print("Expected features:", len(feature_names))
    if new_df_encoded.shape[1] != len(feature_names):
        missing_cols = set(feature_names) - set(new_df_encoded.columns)
        extra_cols = set(new_df_encoded.columns) - set(feature_names)
        print(f"Missing columns: {missing_cols}")
        print(f"Extra columns: {extra_cols}")
        raise ValueError(f"Unexpected input dimension {new_df_encoded.shape[1]}, expected {len(feature_names)}")

except Exception as e:
    print(f"Error during preprocessing: {e}")
    print("Ensure input.csv has compatible columns.")
    exit(1)

try:
    predictions = model.predict(new_df_encoded)
    probabilities = model.predict_proba(new_df_encoded)
except Exception as e:
    print(f"Error during prediction: {e}")
    print(f"Model expects {len(feature_names)} features.")
    exit(1)

print("\nPredictions:")
for i in range(len(predictions)):
    print(f"Row {i}: Prediction = {predictions[i]} (0=Non-Fraud, 1=Fraud), Probability (Non-Fraud, Fraud) = [{probabilities[i][0]:.4f}, {probabilities[i][1]:.4f}]")

print("\nScript completed successfully.")