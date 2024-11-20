import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Load Dataset
data = pd.read_csv("blood_test_dataset.csv")

# Preprocessing: Split Features and Target
X = data.drop(columns=['Disease'])
y = data['Disease']

# Split into Train-Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training (Remove XGBoost, add RandomForest)
models = {
    "Extra Trees": ExtraTreesClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

# Evaluate Models
best_model = None
best_f1 = 0
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"{name} - F1 Score: {f1:.2f}")
    if f1 > best_f1:
        best_f1 = f1
        best_model = model

# Save the Best Model
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Best model saved!")
