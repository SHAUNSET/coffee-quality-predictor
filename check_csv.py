import pandas as pd

# Load your CSV
df = pd.read_csv("data/simulated_coffee_data.csv")

# Quick info about columns, types, missing values
print("DataFrame info:")
print(df.info())

# First few rows
print("\nFirst 5 rows:")
print(df.head())

# Unique values in the Label column
print("\nUnique labels in 'Label' column:")
print(df['Label'].unique())

# Count of missing values per column
print("\nMissing values per column:")
print(df.isnull().sum())
