import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD DATASET
# =========================
app = pd.read_csv("application_record.csv")

print("\nFirst 5 rows:")
print(app.head())

print("\nDataset Info:")
print(app.info())

# =========================
# 1. GENDER DISTRIBUTION
# =========================
print("\nGender Value Counts:")
print(app['CODE_GENDER'].value_counts())

plt.figure()
sns.countplot(x=app['CODE_GENDER'])
plt.title("Gender Distribution")
plt.show()

# =========================
# 2. EDUCATION TYPE
# =========================
print("\nEducation Type Value Counts:")
print(app['NAME_EDUCATION_TYPE'].value_counts())

plt.figure(figsize=(10,5))
sns.countplot(y=app['NAME_EDUCATION_TYPE'])
plt.title("Education Type Distribution")
plt.show()

# =========================
# 3. FAMILY STATUS
# =========================
print("\nFamily Status Value Counts:")
print(app['NAME_FAMILY_STATUS'].value_counts())

plt.figure()
sns.countplot(x=app['NAME_FAMILY_STATUS'])
plt.xticks(rotation=45)
plt.title("Family Status Distribution")
plt.show()

# =========================
# 4. HOUSING TYPE
# =========================
print("\nHousing Type Value Counts:")
print(app['NAME_HOUSING_TYPE'].value_counts())

plt.figure(figsize=(10,5))
sns.countplot(y=app['NAME_HOUSING_TYPE'])
plt.title("Housing Type Distribution")
plt.show()

# =========================
# 5. INCOME ANALYSIS
# =========================
print("\nIncome Statistics:")
print(app['AMT_INCOME_TOTAL'].describe())

plt.figure()
plt.hist(app['AMT_INCOME_TOTAL'], bins=20)
plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Count")
plt.show()

# =========================
# 6. OCCUPATION TYPE
# =========================
print("\nOccupation Type Value Counts:")
print(app['OCCUPATION_TYPE'].value_counts(dropna=False))

plt.figure(figsize=(12,6))
sns.countplot(y=app['OCCUPATION_TYPE'])
plt.title("Occupation Type Distribution")
plt.show()