import pickle
import os
from category_encoders import OneHotEncoder


def check_pkl_contents(encoder_path='encoder.pkl', feature_names_path='feature_names.pkl'):
    """
    Check the contents of encoder.pkl and feature_names.pkl, printing details about saved columns and categories.

    Parameters:
    encoder_path (str): Path to encoder.pkl file.
    feature_names_path (str): Path to feature_names.pkl file.

    Returns:
    None: Prints information about the contents of the pickle files.
    """
    try:
        # Check and load feature_names.pkl
        if not os.path.exists(feature_names_path):
            print(f"Error: {feature_names_path} not found.")
            return
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)
        print(f"\nContents of {feature_names_path}:")
        print(f"Number of features: {len(feature_names)}")
        print("Feature names:")
        for i, col in enumerate(feature_names, 1):
            print(f"  {i}. {col}")

        # Identify categorical feature columns (assuming one-hot encoded columns have a prefix)
        sent_prefix = 'ERC20_most_sent_token_type_'
        rec_prefix = 'ERC20_most_rec_token_type_'
        sent_features = [col for col in feature_names if col.startswith(sent_prefix)]
        rec_features = [col for col in feature_names if col.startswith(rec_prefix)]
        print(f"\nOne-hot encoded features for 'ERC20_most_sent_token_type' ({len(sent_features)}):")
        for col in sent_features:
            print(f"  - {col}")
        print(f"One-hot encoded features for 'ERC20_most_rec_token_type' ({len(rec_features)}):")
        for col in rec_features:
            print(f"  - {col}")

        # Calculate numerical features
        numerical_features = [col for col in feature_names if
                              not (col.startswith(sent_prefix) or col.startswith(rec_prefix))]
        print(f"\nNumerical features ({len(numerical_features)}):")
        for col in numerical_features:
            print(f"  - {col}")

        # Check and load encoder.pkl
        if not os.path.exists(encoder_path):
            print(f"\nError: {encoder_path} not found.")
            return
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)

        # Verify encoder type
        if not isinstance(encoder, OneHotEncoder):
            print(f"\nError: {encoder_path} does not contain a OneHotEncoder object.")
            return

        print(f"\nContents of {encoder_path}:")
        print(f"Encoder type: {type(encoder).__name__}")
        print(f"Categorical columns encoded: {encoder.cols}")
        print(f"Handle unknown categories: {encoder.handle_unknown}")
        print(f"Use category names in output: {encoder.use_cat_names}")

        # Extract categories for each encoded column
        print("\nCategories per encoded column:")
        for col in encoder.cols:
            # Get the mapping of categories to output columns
            mapping = encoder.mapping[col] if isinstance(encoder.mapping, dict) else encoder.mapping
            if isinstance(mapping, list):
                categories = [cat for cat, _ in mapping]
            else:
                categories = mapping.get('mapping', {}).index.tolist()
            print(f"  {col}:")
            for i, cat in enumerate(categories, 1):
                print(f"    {i}. {cat}")

    except Exception as e:
        print(f"\nError reading pickle files: {e}")


if __name__ == "__main__":
    check_pkl_contents()