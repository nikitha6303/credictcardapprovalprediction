import pandas as pd

# Load datasets
application = pd.read_csv("application_record.csv")
credit = pd.read_csv("credit_record.csv")

# Convert STATUS into binary labels
credit["STATUS"] = credit["STATUS"].replace({
    "C": 1,
    "X": 1,
    "0": 1,
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0
})

# Merge the datasets
data = application.merge(credit, on="ID", how="inner")

# Display the merged dataset
print(data.head())

# Display dataset size
print(data.shape)