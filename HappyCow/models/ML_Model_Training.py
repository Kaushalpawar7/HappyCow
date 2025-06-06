import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

def load_dataset(dataset_path='cow_health_dataset_realistic_v5.xlsx', new_data_path=None):
    df = pd.read_excel(dataset_path)
    if new_data_path and os.path.exists(new_data_path):
        new_df = pd.read_excel(new_data_path)
        if not all(col in new_df.columns for col in df.columns):
            raise ValueError("New data missing required columns")
        new_df = new_df.dropna()
        df = pd.concat([df, new_df], ignore_index=True)
        print(f"Appended {len(new_df)} new samples. Total samples: {len(df)}")
    return df

def train_model(df, model_path='cow_health_classifier.pkl'):
    features = ['Temperature (°C)', 'BPM', 'Activity', 'Temp_Trend', 'BPM_Trend', 'Activity_Trend']
    X = df[features]
    y = df[['Fever', 'Mastitis', 'Black Quarter', 'Bovine Pneumonia', 'Anaplasmosis', 'Blue Tongue']]

    print("Positive cases per disease:")
    for disease in y.columns:
        positive_count = y[disease].sum()
        print(f"{disease}: {positive_count}")
        if positive_count < 2:
            print(f"Warning: {disease} has fewer than 2 positive cases.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Using random split (multi-label stratification not supported).")

    class_weights = {
        'Fever': {0: 1, 1: 1},
        'Mastitis': {0: 1, 1: 5},
        'Black Quarter': {0: 1, 1: 10},
        'Bovine Pneumonia': {0: 1, 1: 7},
        'Anaplasmosis': {0: 1, 1: 10},
        'Blue Tongue': {0: 1, 1: 12}
    }
    base_rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    classifier = MultiOutputClassifier(base_rf, n_jobs=-1)

    if not os.path.exists(model_path):
        param_grid = {
            'estimator__n_estimators': [100, 200],
            'estimator__max_depth': [10, 20, None],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__min_samples_leaf': [1, 2]
        }
        grid_search = GridSearchCV(
            classifier, param_grid, cv=5, scoring='f1_macro', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        classifier = grid_search.best_estimator_
        print("Best parameters:", grid_search.best_params_)
        
        for i, estimator in enumerate(classifier.estimators_):
            disease = y.columns[i]
            estimator.class_weight = class_weights[disease]
        classifier.fit(X_train, y_train)
    else:
        classifier.fit(X_train, y_train)

    # Optimize thresholds
    y_pred_proba = classifier.predict_proba(X_test)
    thresholds = {
        'Fever': 0.5,
        'Mastitis': 0.4,
        'Black Quarter': 0.45,
        'Bovine Pneumonia': 0.35,
        'Anaplasmosis': 0.5,
        'Blue Tongue': 0.4
    }
    y_pred = []
    for i, disease in enumerate(y.columns):
        threshold = thresholds[disease]
        pred = (y_pred_proba[i][:, 1] > threshold).astype(int)
        y_pred.append(pred)
    y_pred = np.array(y_pred).T

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Macro Precision: {precision:.2f}")
    print(f"Macro Recall: {recall:.2f}")
    print(f"Macro F1-score: {f1:.2f}")

    for i, disease in enumerate(y.columns):
        print(f"\nMetrics for {disease}:")
        print(f"Precision: {precision_score(y_test[disease], y_pred[:, i], zero_division=0):.2f}")
        print(f"Recall: {recall_score(y_test[disease], y_pred[:, i], zero_division=0):.2f}")
        print(f"F1-score: {f1_score(y_test[disease], y_pred[:, i], zero_division=0):.2f}")

    with open(model_path, 'wb') as file:
        pickle.dump(classifier, file)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    new_data_path = None
    df = load_dataset(new_data_path=new_data_path)
    train_model(df)