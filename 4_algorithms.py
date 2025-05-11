# 4_algorithms.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from check import predict_fraud

# 1. Dataset acquisition and exploratory data analysis
df = pd.read_csv('upi_fraud_dataset.csv')

print("Dataset head:\n", df.head())
print("\nDataset info:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nClass distribution:\n", df['is_fraud'].value_counts())

# Summarize dataset
print("\nStatistical summary:")
print(df.describe())

# 2. Data preprocessing and feature engineering

# Convert transaction_date to ordinal
df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
df = df.dropna(subset=['transaction_date'])  # drop rows with invalid dates
df['transaction_date_ordinal'] = df['transaction_date'].map(datetime.toordinal)

# Validate transaction_id and utr_number format (basic filtering)
def valid_transaction_id(txn_id):
    return isinstance(txn_id, str) and len(txn_id) == 23 and txn_id[0] == 'T' and txn_id[1:].isdigit()

def valid_utr_number(utr):
    return isinstance(utr, str) and len(utr) == 12 and utr.isdigit()

df = df[df['transaction_id'].apply(valid_transaction_id)]
df = df[df['utr_number'].apply(valid_utr_number)]

print(f"\nDataset size after validation: {len(df)}")

# Encode categorical variables
le_receiver = LabelEncoder()
le_txn = LabelEncoder()
le_utr = LabelEncoder()

df['receiver_upi_enc'] = le_receiver.fit_transform(df['receiver_upi_id'])
df['transaction_id_enc'] = le_txn.fit_transform(df['transaction_id'])
df['utr_number_enc'] = le_utr.fit_transform(df['utr_number'])

# Features and target
X = df[['transaction_date_ordinal', 'transaction_amount', 'receiver_upi_enc', 'transaction_id_enc', 'utr_number_enc']]
y = df['is_fraud']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 4. Implement foundational ML algorithms and evaluate

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

# 5. Deep Learning Model

dl_model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dl_model.fit(X_train, y_train, epochs=30, batch_size=8, verbose=1, validation_split=0.1)

# Evaluate DL model
loss, accuracy = dl_model.evaluate(X_test, y_test, verbose=0)
print(f"\nDeep Learning Model Accuracy: {accuracy:.4f}")

# 6. Model selection and parameter tuning (example for Random Forest)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)
print(f"\nBest Random Forest params: {grid.best_params_}")
print(f"Best RF CV accuracy: {grid.best_score_:.4f}")

best_rf = grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)
print("Random Forest after tuning accuracy:", accuracy_score(y_test, y_pred_rf))

# 7. Cross-validation on best model
cv_scores = cross_val_score(best_rf, X_scaled, y, cv=5)
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f}")

# 8. Save models and encoders

pickle.dump(best_rf, open('random_forest_model.pkl', 'wb'))
pickle.dump(le_receiver, open('le_receiver_upi.pkl', 'wb'))
pickle.dump(le_txn, open('le_transaction_id.pkl', 'wb'))
pickle.dump(le_utr, open('le_utr_number.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

dl_model.save('deep_learning_model.h5')

print("\nModels and encoders saved successfully.")