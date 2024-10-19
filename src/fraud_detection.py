# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Load dataset
import os
file_path = os.path.join(os.path.dirname(__file__), '../data/fraud_data.csv')
data = pd.read_csv(file_path)


# Initial data exploration
print("Dataset shape:", data.shape)
print("First few rows:\n", data.head())

# Data Preprocessing
# Handling missing values - Example (Impute with mean for simplicity)
data.fillna(data.mean(), inplace=True)

# Feature engineering (dummy example)
# Assuming 'Amount', 'Time' are present in the data, others might vary
X = data.drop(columns=['isFraud'])  # Drop target variable (isFraud)
y = data['isFraud']

# Data scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(sampling_strategy='minority')
X_res, y_res = smote.fit_resample(X_scaled, y)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Model Building - Ensemble Methods

# 1. Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# 2. Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# 3. XGBoost Classifier
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

# Model Evaluation

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("AUC-ROC Score:", roc_auc_score(y_test, y_prob))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_prob):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()

# Evaluate each model
print("Random Forest Model Performance:")
evaluate_model(rf_model, X_test, y_test)

print("Gradient Boosting Model Performance:")
evaluate_model(gb_model, X_test, y_test)

print("XGBoost Model Performance:")
evaluate_model(xgb_model, X_test, y_test)

# Hyperparameter Tuning for XGBoost (Optional)
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200]
}
grid_search = GridSearchCV(XGBClassifier(random_state=42), param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

print("Best parameters from Grid Search:", grid_search.best_params_)
