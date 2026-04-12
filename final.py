# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np

from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import friedmanchisquare

# 2. LOAD DATASET

print("\n*...Loading dataset...*")

data = pd.read_csv("CIC-AndMal2017_preprocess_v2.csv")

# Reduce dataset size 
data = data.sample(n=5000, random_state=42)

data = data.dropna()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print(f"Dataset Loaded : {X.shape[0]} samples, {X.shape[1]} features")

# 3. TRAIN-TEST SPLIT

print("\n*...Splitting dataset...*")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training Samples: {X_train.shape[0]}")
print(f"Testing Samples : {X_test.shape[0]}")

# 4. SCALING

print("\n*...Scaling features...*")

scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# 5. FEATURE SELECTION

print("\n*...Applying Feature Selection Techniques...*")

# Information Gain
ig = mutual_info_classif(X_train, y_train)

# Chi-Square
chi, _ = chi2(X_train, y_train)

# Extra Trees Classifier
et_model = ExtraTreesClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
et_model.fit(X_train, y_train)
et = et_model.feature_importances_

# Mean Term Frequency
tf = X_train.sum(axis=0).values

# Inverse Document Frequency
idf = np.log((len(X_train) + 1) / (X_train.sum(axis=0) + 1)).values

# MTF-IDF
tfidf = tf * idf

print("\n*...Feature scoring completed...*")

# 6. RANKING

print("\n*...Ranking features...*")

def rank_features(scores):
    return np.argsort(np.argsort(-scores))

ig_rank = rank_features(ig)
chi_rank = rank_features(chi)
et_rank = rank_features(et)
tf_rank = rank_features(tf)
idf_rank = rank_features(idf)
tfidf_rank = rank_features(tfidf)

# 7. FRIEDMAN TEST

stat, p = friedmanchisquare(
    ig_rank, chi_rank, et_rank, tf_rank, idf_rank, tfidf_rank
)

print(f"Friedman Test p-value: {p:.4f}")

# 8. RANK FUSION

print("\n*...Combining rankings (Rank Fusion)...*")

rank_matrix = np.vstack([
    ig_rank,
    chi_rank,
    et_rank,
    tf_rank,
    idf_rank,
    tfidf_rank
]).T

final_rank = rank_matrix.sum(axis=1)

# 9. SELECT TOP FEATURES

top_n = 50
top_indices = np.argsort(final_rank)[:top_n]

selected_features = X_train.columns[top_indices]

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

print(f"\n*...Top {top_n} Features Selected...*")

# 10. MODEL TRAINING

print("\n*...Training Random Forest model...*")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_selected, y_train)

print("\n*...Model training completed...*")

# 11. EVALUATION 

print("\n*...Evaluating model...*")

y_pred = model.predict(X_test_selected)

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print("MODEL PERFORMANCE SUMMARY :")

print(f"\nAccuracy: {acc:.2f}")

print("\nClass-wise Performance :")

print("\nClass 0 (Benign):")
print(f"  Precision: {report['0']['precision']:.2f}")
print(f"  Recall   : {report['0']['recall']:.2f}")
print(f"  F1-Score : {report['0']['f1-score']:.2f}")

print("\nClass 1 (Malware):")
print(f"  Precision: {report['1']['precision']:.2f}")
print(f"  Recall   : {report['1']['recall']:.2f}")
print(f"  F1-Score : {report['1']['f1-score']:.2f}")

print("\nOverall Performance:")
print(f"  Macro Avg     : {report['macro avg']['f1-score']:.2f}")
print(f"  Weighted Avg  : {report['weighted avg']['f1-score']:.2f}")