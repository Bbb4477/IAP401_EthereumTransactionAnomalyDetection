import pickle
import os


def check_model_columns(model_path='XGBoost.pkl', feature_names_path='feature_names.pkl'):
    """
    Check the number of columns (features) expected by the model by reading XGBoost.pkl
    and feature_names.pkl, and verify consistency between them.

    Parameters:
    model_path (str): Path to XGBoost.pkl file.
    feature_names_path (str): Path to feature_names.pkl file.

    Returns:
    None: Prints the number of columns from both files and the feature names.
    """
    try:
        # Check if files exist
        if not os.path.exists(model_path):
            print(f"Error: {model_path} not found.")
            return
        if not os.path.exists(feature_names_path):
            print(f"Error: {feature_names_path} not found.")
            return

        # Load the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load feature names
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)

        # Get number of features from the model
        try:
            model_features = model.n_features_in_
            print(f"The model (XGBoost.pkl) expects {model_features} columns.")
        except AttributeError:
            print(
                "Error: Unable to retrieve number of features from XGBoost.pkl. Ensure it is a valid XGBClassifier model.")
            return

        # Get number of features from feature_names.pkl
        feature_names_count = len(feature_names)
        print(f"The feature_names.pkl expects {feature_names_count} columns.")

        # Check for consistency
        if model_features != feature_names_count:
            print(
                f"Warning: Mismatch detected! Model expects {model_features} columns, but feature_names.pkl has {feature_names_count} columns.")

        # Print feature names for reference
        print("\nExpected feature columns from feature_names.pkl:")
        for i, col in enumerate(feature_names, 1):
            print(f"  {i}. {col}")

        # Additional check: Verify feature_names.pkl is a list
        if not isinstance(feature_names, list):
            print("Error: feature_names.pkl does not contain a list of feature names.")

    except Exception as e:
        print(f"Error reading files: {e}")
        print("Ensure XGBoost.pkl and feature_names.pkl are valid pickle files in the correct directory.")


if __name__ == "__main__":
    check_model_columns()