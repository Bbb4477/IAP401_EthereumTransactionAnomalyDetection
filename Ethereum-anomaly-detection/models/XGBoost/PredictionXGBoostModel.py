import pandas as pd
import pickle
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ============================
# 1. Load Dataset
# ============================
df = pd.read_csv("transaction_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace(r'\b\s+\b', '_', regex=True)

# Drop irrelevant columns (adjust axis=1 for columns)
df.drop(columns=['Unnamed: 0', 'Index', 'Address'], errors='ignore', inplace=True)

# Handle missing values
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

categorical_cols = ['ERC20_most_sent_token_type', 'ERC20_most_rec_token_type']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# Separate features/target
if "FLAG" in df.columns:
    X = df.drop(columns="FLAG")
    y = df["FLAG"]
else:
    raise ValueError("❌ The dataset does not have a 'FLAG' column")

# ============================
# 2. Preprocessing (same as training)
# ============================
encoder = TargetEncoder(cols=categorical_cols)
X_encoded = encoder.fit_transform(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# ============================
# 3. Load Trained Model
# ============================
model = pickle.load(open("XGBoost.pkl", "rb"))

# ============================
# 4. Prediction Loop
# ============================
for i in range(0, 100):  # first 100 rows (adjust as needed)
    example_row = X_scaled[i].reshape(1, -1)

    prediction = model.predict(example_row)[0]
    prob = model.predict_proba(example_row)[0]
    true_label = y.iloc[i]

    print(f"Index: {i}")
    print(f"Prediction: {'Fraud (1)' if prediction == 1 else 'Non-Fraud (0)'}")
    print(f"Probability (Non-Fraud, Fraud): [{prob[0]:.4f}, {prob[1]:.4f}]")
    print(f"True label (FLAG): {true_label} (1=fraud, 0=non-fraud)")
    print("--------------------------------------------------")
