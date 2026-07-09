import pandas as pd

# Load dataset
df = pd.read_csv("credit_record.csv")

# Display first five rows
print(df.head())

# Display dataset information
print(df.info())

# Display descriptive statistics
print(df.describe())