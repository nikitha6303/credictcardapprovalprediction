import pandas as pd
data = {
'ID': [1,2,3,4,5,6,7,8],
'CODE_GENDER': ['M','F','F','M','M','F','M','F'],
'FLAG_OWN_CAR': ['Y','N','Y','Y','N','N','Y','N'],
'CNT_CHILDREN': [0,1,0,2,1,0,3,1],
'AMT_INCOME_TOTAL': [50000,60000,45000,70000,80000,52000,61000,48000],
'OCCUPATION_TYPE': ['Labor','Manager',None,'Clerk',None,'Sales',None,None]
}
df = pd.DataFrame(data)
print("Missing Values Count:")
print(df.isnull().sum())
print("\nMissing Values Percentage:")
print(df.isnull().mean())
df = df.drop('OCCUPATION_TYPE', axis=1)
print("\nAfter Removing Column:")
print(df.isnull().sum())
df['CODE_GENDER'] = df['CODE_GENDER'].map({'M':1, 'F':0})
df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].map({'Y':1, 'N':0})
X = df[['AMT_INCOME_TOTAL', 'CNT_CHILDREN', 'CODE_GENDER']]
y = df['FLAG_OWN_CAR']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("\nPredictions:", predictions)
new_data = pd.DataFrame([[55000, 1, 1]],
columns=['AMT_INCOME_TOTAL','CNT_CHILDREN','CODE_GENDER'])
print("\nPrediction for new data:", model.predict(new_data))