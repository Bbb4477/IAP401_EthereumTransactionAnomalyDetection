import pandas as pd
df = pd.read_csv('transaction_dataset.csv')
print(df.columns.tolist())
print(len(df.columns.tolist()))