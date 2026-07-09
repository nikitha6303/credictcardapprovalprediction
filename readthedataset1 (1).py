import pandas as pd
data = {
'ID': [1,2,3,4,5,6,7,8],
'CODE_GENDER': ['M','F','F','M','M','F','M','F'],
'AMT_INCOME_TOTAL': [50000,60000,45000,70000,80000,52000,61000,48000],
'FLAG_OWN_CAR': ['Y','N','Y','Y','N','N','Y','N']
}
app = pd.DataFrame(data)
print(" First 5 Rows:")
print(app.head())
print("\n Dataset Shape:")
print(app.shape)
print("\n Dataset Info:")
app.info()
print("\n Missing Values:")
print(app.isnull().sum())
app['CODE_GENDER'] = app['CODE_GENDER'].map({'M':1, 'F':0})
app['FLAG_OWN_CAR'] = app['FLAG_OWN_CAR'].map({'Y':1, 'N':0})
X = app[['AMT_INCOME_TOTAL']]
y = app['FLAG_OWN_CAR']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("\n Predictions:")
print(predictions)
from sklearn.metrics import accuracy_score
print("\n Accuracy:")
print(accuracy_score(y_test, predictions))
new_data = pd.DataFrame([[55000]], columns=['AMT_INCOME_TOTAL'])
prediction = model.predict(new_data)
print("\n Prediction for income = 55000:")
print(prediction)