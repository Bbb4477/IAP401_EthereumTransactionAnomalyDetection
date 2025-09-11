import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import pickle
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
import os

for dirname, _, filenames in os.walk('.'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("transaction_dataset.csv")
df.columns = df.columns.str.strip().str.replace(r'\b\s+\b', '_', regex=True)
df.drop(columns=['Unnamed: 0', 'Index', 'Address'], axis=0, inplace=True)
categories = df.select_dtypes(include=['object']).columns
numeric = df.select_dtypes(include=['number']).columns
constant_var = [i for i in numeric if df[i].var() == 0]
df.drop(columns=constant_var, axis=0, inplace=True)
df_imputed = df.copy()
numeric_cols = df_imputed.select_dtypes(include=['number']).columns
df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(0)
categorical_cols = ['ERC20_most_sent_token_type', 'ERC20_most_rec_token_type']
df_imputed[categorical_cols] = df_imputed[categorical_cols].fillna('Unknown')
# Split data
XT = df_imputed.drop(columns='FLAG')
yT = df_imputed['FLAG']
# Encode categorical columns
encoder = TargetEncoder(cols=categorical_cols)
XTe = encoder.fit_transform(XT, yT)
# Scale features (important for Logistic Regression)
scaler = StandardScaler()
XTs = scaler.fit_transform(XTe)


model = pickle.load(open('model.pkl','rb'))

# Prediction in range
# for i in range(7000,8500):
#     example_row_scaled = XTs[i].reshape(1, -1)  # Use scaled features
#     prediction = model.predict(example_row_scaled)
#     prob = model.predict_proba(example_row_scaled)[0]  # Probability of each class
#     true_label = yT.iloc[i]
#     print(f"Index: {i}")
#     print(f"Prediction: {'Fraud (1)' if prediction[0] == 1 else 'Non-Fraud (0)'}")
#     print(f"Probability (Non-Fraud, Fraud): [{prob[0]:.4f}, {prob[1]:.4f}]")
#     print(f"True label (FLAG): {true_label} (1=fraud, 0=non-fraud)")
#     # print(XT.iloc[i])
#     print("\n")

# Single prediction
i = 8048  # Change this to select input from dataset.csv
example_row_scaled = XTs[i].reshape(1, -1)  # Use scaled features
prediction = model.predict(example_row_scaled)
prob = model.predict_proba(example_row_scaled)[0]  # Probability of each class
true_label = yT.iloc[i]
print(f"Index: {i}")
print(f"Prediction: {'Fraud (1)' if prediction[0] == 1 else 'Non-Fraud (0)'}")
print(f"Probability (Non-Fraud, Fraud): [{prob[0]:.4f}, {prob[1]:.4f}]")
print(f"True label (FLAG): {true_label} (1=fraud, 0=non-fraud)")
print()
print(XT.iloc[i])
print("")