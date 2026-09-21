# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset

The aim is to predict daily temperature, PM2.5 pollution level, and Energy using environmental sensor data and the Random Forest Algorithm. The dataset used is **`weather-station-eee-block_2024_07_13.csv`**, which contains various environmental sensor readings.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load and preprocess the weather sensor data.
2. Extract relevant environmental and time features.
3. Split the data into training and testing sets.
4. Train Random Forest models for temperature, PM2.5, and energy.
5. Predict the outputs and calculate MAE, RMSE, and R².
 
## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: P.Pramisha
RegisterNumber: 212224230203
*/
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("weather.csv")


df["time"] = pd.to_datetime(df["time"])

df = df.sort_values("time").reset_index(drop=True)


df["hour"] = df["time"].dt.hour
df["dayofyear"] = df["time"].dt.dayofyear
df["month"] = df["time"].dt.month
df["dayofweek"] = df["time"].dt.dayofweek


df = df.dropna(subset=["tem"])


features = [
    "hum",
    "co2",
    "illumination",
    "pressure",
    "pm2_5",
    "pm10",
    "wind_direction",
    "wind_direction_angle",
    "wind_speed",
    "wind_speed_level",
    "tsr",
    "hour",
    "dayofyear",
    "month",
    "dayofweek"
]

X = df[features]
y = df["tem"]


numeric_features = [
    "hum", "co2", "illumination", "pressure",
    "pm2_5", "pm10", "wind_direction_angle",
    "wind_speed", "wind_speed_level", "tsr",
    "hour", "dayofyear", "month", "dayofweek"
]

categorical_features = ["wind_direction"]


preprocessor = ColumnTransformer([
    (
        "num",
        SimpleImputer(strategy="median"),
        numeric_features
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_features
    )
])


rf = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("random_forest", rf)
])


split = int(len(X) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Random Forest Results")
print("---------------------")
print("MAE  :", mae)
print("RMSE :", rmse)
print("R²   :", r2)
```

## Output:

<img width="591" height="108" alt="image" src="https://github.com/user-attachments/assets/9cc61061-fb05-4239-9548-1cd350549d76" />

## Result:
Thus, the Random Forest Algorithm was successfully applied to analyze environmental sensor data for predicting daily temperature, PM2.5 pollution levels, and energy consumption.
