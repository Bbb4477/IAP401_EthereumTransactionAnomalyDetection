import pandas as pd
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

# Function to load files with error handling
def load_pickle_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)

try:
    # Load the pre-trained model, encoder, and feature names
    model = load_pickle_file('XGBoost.pkl')
    encoder = load_pickle_file('encoder.pkl')
    feature_names = load_pickle_file('feature_names.pkl')
except FileNotFoundError as e:
    print(e)
    print("Ensure 'XGBoost.pkl', 'encoder.pkl', and 'feature_names.pkl' are in './XGBoostTest/'.")
    print("To generate 'encoder.pkl' and 'feature_names.pkl', add the following to the training notebook after encoding and before training:")
    print("pickle.dump(encoder, open('./XGBoostTest/encoder.pkl', 'wb'))")
    print("X = df.drop('FLAG', axis=1)")
    print("pickle.dump(list(X.columns), open('./XGBoostTest/feature_names.pkl', 'wb'))")
    exit(1)
except Exception as e:
    print(f"Error loading files: {e}")
    exit(1)

# Print expected columns from feature_names.pkl
print(f"Expected features ({len(feature_names)}):")
for i, col in enumerate(feature_names, 1):
    print(f"  {i}. {col}")

# Load input data
try:
    new_df = pd.read_csv('input.csv')
except FileNotFoundError:
    print("Input file 'input.csv' not found. Ensure it exists with columns matching the original dataset.")
    exit(1)
except Exception as e:
    print(f"Error reading input.csv: {e}")
    exit(1)

# Preprocessing steps to match training
try:
    # Debug: Print input columns
    print("\nInput CSV columns:", list(new_df.columns))

    # Standardize column names by replacing spaces with underscores
    new_df.columns = [col.strip().replace(' ', '_') for col in new_df.columns]

    # Debug: Print columns after standardization
    print("Columns after standardization:", list(new_df.columns))

    # Drop irrelevant columns if present
    columns_to_drop = ['Unnamed:_0', 'Index', 'Address']
    new_df.drop([col for col in columns_to_drop if col in new_df.columns], axis=1, inplace=True)

    # Drop 'FLAG' if present
    if 'FLAG' in new_df.columns:
        new_df.drop('FLAG', axis=1, inplace=True)

    # Define categorical and numerical features
    cat_features = ['ERC20_most_sent_token_type', 'ERC20_most_rec_token_type']
    num_features = [col for col in new_df.columns if col not in cat_features]

    # Debug: Print feature counts
    print(f"Numerical features ({len(num_features)}):", num_features)
    print(f"Categorical features ({len(cat_features)}):", cat_features)

    # Fill NaNs
    new_df[num_features] = new_df[num_features].fillna(0)
    new_df[cat_features] = new_df[cat_features].fillna('')

    # Apply the saved encoder
    new_df_encoded = encoder.transform(new_df)

    # Debug: Print encoded columns
    print("Columns after encoding:", list(new_df_encoded.columns))
    print("Encoded shape:", new_df_encoded.shape)

    # Fill NaNs from unknown categories with 0
    new_df_encoded = new_df_encoded.fillna(0)

    # Align features to match the model's expected columns
    new_df_encoded = new_df_encoded.reindex(columns=feature_names, fill_value=0)

    # Debug: Check final shape and column mismatches
    print("Final encoded shape:", new_df_encoded.shape)
    print("Expected features:", len(feature_names))
    if new_df_encoded.shape[1] != len(feature_names):
        missing_cols = set(feature_names) - set(new_df_encoded.columns)
        extra_cols = set(new_df_encoded.columns) - set(feature_names)
        print(f"Missing columns: {missing_cols}")
        print(f"Extra columns: {extra_cols}")
        raise ValueError(f"Unexpected input dimension {new_df_encoded.shape[1]}, expected {len(feature_names)}")

    # Ensure no extra columns
    extra_cols = set(new_df_encoded.columns) - set(feature_names)
    if extra_cols:
        print(f"Warning: Dropping extra columns not in model: {extra_cols}")
        new_df_encoded.drop(extra_cols, axis=1, inplace=True)
except Exception as e:
    print(f"Error during preprocessing: {e}")
    print("Ensure input.csv has compatible columns (e.g., no new numerical columns without handling).")
    exit(1)

# Make predictions
try:
    predictions = model.predict(new_df_encoded)
    probabilities = model.predict_proba(new_df_encoded)
except Exception as e:
    print(f"Error during prediction: {e}")
    print(f"Possible mismatch in feature count or types. Model expects {len(feature_names)} features.")
    exit(1)

# Output results
print("\nPredictions:")
for i in range(len(predictions)):
    print(f"Row {i}: Prediction = {predictions[i]} (0=Non-Fraud, 1=Fraud), Probability (Non-Fraud, Fraud) = [{probabilities[i][0]:.4f}, {probabilities[i][1]:.4f}]")

print("\nScript completed successfully.")