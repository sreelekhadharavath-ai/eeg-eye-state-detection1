import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler

print("=== Starting Advanced EEG Eye State Pipeline ===\n")

data_file = 'eeg_data.csv'
if not os.path.exists(data_file):
    print("Error: Dataset not found. Please run download_dataset.py first.")
    exit()

df = pd.read_csv(data_file)
print("Dataset loaded. Preprocessing...")
df = df.dropna()

def add_engineered_features(data):
    features = data.copy()
    channel_cols = features.columns[:14]
    features['mean_amp'] = features[channel_cols].mean(axis=1)
    features['std_amp'] = features[channel_cols].std(axis=1)
    features['max_amp'] = features[channel_cols].max(axis=1)
    features['min_amp'] = features[channel_cols].min(axis=1)
    return features

target_col = 'eye_state'
X = df.drop(columns=[target_col])
y = df[target_col]

print("Engineering Spatial Features...")
X_engineered = add_engineered_features(X)

print("Scaling with RobustScaler (resistant to eye-blink outliers)...")
scaler = RobustScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_engineered), columns=X_engineered.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

print("\n--- Training Smaller ExtraTrees Classifier --- ")
print("(Optimized for smaller file size to fit GitHub's 25MB upload limit)")
model = ExtraTreesClassifier(n_estimators=30, max_depth=15, min_samples_split=2, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

y_predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, y_predictions)
print(f"\n>>> NEW MODEL ACCURACY: {accuracy * 100:.2f}% <<<")

print("Classification Report:")
print(classification_report(y_test, y_predictions, target_names=["Open (0)", "Closed (1)"]))

print("\n--- Saving Improved Model & Scaler ---")
joblib.dump(model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Saved successfully!")
