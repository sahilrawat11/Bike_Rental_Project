#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from xgboost import XGBRegressor

# ----------------- LOGGER SETUP -----------------
logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

# ----------------- DATA LOADING -----------------
def load_data(file_path):
    df = pd.read_csv("BikeRentalData.csv")
    # Remove columns not useful for prediction
    drop_cols = [ "yr","holiday","workingday","atemp" ,  "registered"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    logging.info("✅ Data loaded and cleaned")
    return df

# ----------------- ENCODING -----------------
def encode_data(df, target_col="cnt"):
    te = TargetEncoder(cols=['mnth', 'hr', 'weekday'])
    df[['mnth','hr','weekday']] = te.fit_transform(df[['mnth','hr','weekday']], df[target_col])
    logging.info("✅ Data encoded")
    return df, te

# ----------------- TRAIN-TEST SPLIT -----------------
def split_data(df, target_col="cnt"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------- SCALING -----------------
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# ----------------- MODEL TRAINING (WITH GRIDSEARCH) -----------------
def train_model(X_train, y_train):
    xgb = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='r2', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    logging.info(f"✅ Best Params: {grid_search.best_params_}")
    logging.info("✅ Model training complete")
    return grid_search.best_estimator_

# ----------------- EVALUATION -----------------
def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    logging.info(f"📊 Train R² Score: {train_r2:.4f} | Train MAE: {train_mae:.2f}")
    logging.info(f"📊 Test R² Score: {test_r2:.4f} | Test MAE: {test_mae:.2f}")

    return train_r2, test_r2

# ----------------- SAVE OBJECTS -----------------
def save_objects(model, scaler, encoder):
    joblib.dump(model, "trained_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoder, "encoder.pkl")
    logging.info("✅ Saved model, scaler, and encoder")

# ----------------- MAIN SCRIPT -----------------
if __name__ == "__main__":
    df = load_data("BikeRentalData.csv")
    df, encoder = encode_data(df, target_col="cnt")

    X_train, X_test, y_train, y_test = split_data(df, target_col="cnt")
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    model = train_model(X_train_scaled, y_train)
    evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)

    save_objects(model, scaler, encoder)


# In[ ]:




