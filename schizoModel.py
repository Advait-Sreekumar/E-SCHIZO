import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import classification_report, accuracy_score, recall_score

# 1. Prepare Data
df = pd.read_csv('schizophrenia_training_data_v2.csv')
# The "Perfection" Feature: Inhibition Efficiency Index
df['Efficiency_Index'] = df['Mean_RT'] / (1.0 - df['Error_Rate'])

features = ['Mean_RT', 'CV_RT', 'SSRT', 'Skew_RT', 'Efficiency_Index']
X = df[features].values
y = df['Label'].map({'CONTROL': 0, 'SCHZ': 1}).values

# 2. Power Transformation
scaler = PowerTransformer(method='yeo-johnson')
X_scaled = scaler.fit_transform(X)

# 3. Best Stability Ensemble
clf_rf = RandomForestClassifier(n_estimators=500, max_depth=3, min_samples_leaf=15, class_weight='balanced', random_state=42)
clf_svm = SVC(kernel='rbf', C=0.6, class_weight='balanced', probability=True, random_state=42)
ensemble = VotingClassifier(estimators=[('rf', clf_rf), ('svm', clf_svm)], voting='soft')

# 4. Stratified Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
ensemble.fit(X_train, y_train)

# 5. Threshold Optimization (0.38)
final_threshold = 0.38 

# Predictions for Training Set
train_probs = ensemble.predict_proba(X_train)[:, 1]
y_pred_train = (train_probs >= final_threshold).astype(int)

# Predictions for Testing Set
test_probs = ensemble.predict_proba(X_test)[:, 1]
y_pred_test = (test_probs >= final_threshold).astype(int)

# 6. Final Performance Check
cv_results = cross_validate(ensemble, X_scaled, y, cv=StratifiedKFold(10, shuffle=True, random_state=42), scoring='accuracy')

print("="*50)
# print("🏥 FINALIZED BIOMARKER RESEARCH MODEL")
# print("="*50)
# ADDED: Training Accuracy for comparison
print(f"Training Accuracy:     {accuracy_score(y_train, y_pred_train)*100:.2f}%")
print(f"Testing Accuracy:      {accuracy_score(y_test, y_pred_test)*100:.2f}%")
print(f"Schiz Recall:          {recall_score(y_test, y_pred_test)*100:.2f}%")
print(f"10-Fold CV Mean:       {np.mean(cv_results['test_score'])*100:.2f}%")
print(f"Confidence Interval:   +/- {np.std(cv_results['test_score'])*2*100:.2f}%")
print("-" * 50)
print(classification_report(y_test, y_pred_test, target_names=['CONTROL', 'SCHZ']))

# 7. Save for your App
joblib.dump(ensemble, 'schizo_model.pkl')
joblib.dump(scaler, 'schizo_scaler.pkl')
print("Model and Scaler Saved.")