# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# =========================
# 2. Load Dataset
# =========================
data = pd.read_csv("D:/University work/Linkdin Projects/german_credit_data.csv")

# =========================
# 3. Fix TARGET column
# =========================
# Convert good/bad → 1/0
data['target'] = data['target'].map({'good': 1, 'bad': 0})

# =========================
# 4. Handle Missing Values
# =========================
data = data.dropna()

# =========================
# 5. Convert Categorical → Numeric
# =========================
data = pd.get_dummies(data, drop_first=True)

# =========================
# 6. Split Features & Target
# =========================
X = data.drop('target', axis=1)
y = data['target']

# =========================
# 7. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 8. Feature Scaling
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 9. Models
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# =========================
# 10. Training & Evaluation
# =========================
for name, model in models.items():
    print("\n=========================")
    print(f"Model: {name}")
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("ROC-AUC Score:")
    print(roc_auc_score(y_test, y_prob))

# =========================
# 11. Save Model
# =========================
import joblib
joblib.dump(model, "credit_model.pkl")

print("\nModel Saved Successfully!")