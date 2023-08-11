import pandas as pd
import numpy as np

from sklearn.metrics import explained_variance_score as evs
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


# Import Data
df = pd.read_csv('Shoe prices.csv')


# EDA & Preprocessing
print(df.info())

print(df['Price (USD)'])

for value in df['Price (USD)'].values:
    df['Price (USD)'] = df['Price (USD)'].replace(value, value[1::])

df['Price (USD)'] = df['Price (USD)'].astype('float')

print(df['Price (USD)'])

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
print(df.info())


# Train Test Split
features = df.drop('Price (USD)', axis= 1)
target = df['Price (USD)']

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size= 0.25, random_state= 42)


# Model Training
models = [XGBRegressor(),
          RandomForestRegressor(),
          GradientBoostingRegressor(),
          DecisionTreeRegressor(),
          ]

for m in models:
    print(m)

    m.fit(X_train, Y_train)

    pred_train = m.predict(X_train)
    print(f'Training Accuracy is :{evs(Y_train, pred_train)}')

    pred_test = m.predict(X_test)
    print(f'Test Accuracy is :{evs(Y_test, pred_test)}\n')