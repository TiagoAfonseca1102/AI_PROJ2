import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)
import matplotlib.pyplot as plt
from data.generate_data import generate_appointments

# ── Dados 
df = generate_appointments(n=2000)

FEATURES_NUM = ["age", "distance_km", "previous_noshow_rate",
                "lead_days", "appointment_hour"]
FEATURES_CAT = ["gender", "specialty"]
FEATURES_BIN = ["chronic_disease", "sms_received", "is_first_visit", "day_of_week"]
ALL_FEATURES  = FEATURES_NUM + FEATURES_CAT + FEATURES_BIN

X = df[ALL_FEATURES]
y = df["no_show"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pré-processamento 
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_CAT),
    ("bin", "passthrough", FEATURES_BIN),
])

# Modelos candidatos 
candidates = {
    "Logistic Regression":      LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":            RandomForestClassifier(n_estimators=150, random_state=42),
    "Gradient Boosting":        GradientBoostingClassifier(n_estimators=150, random_state=42),
}

print(" Cross-validation (ROC-AUC)")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_name, best_score, best_clf = None, 0, None

for name, clf in candidates.items():
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    mean_auc = scores.mean()
    print(f"  {name:30s}: {mean_auc:.4f} ± {scores.std():.4f}")
    if mean_auc > best_score:
        best_score, best_name, best_clf = mean_auc, name, clf

# Treino  
print(f"\nMelhor modelo: {best_name} (AUC={best_score:.4f})")
pipeline = Pipeline([("pre", preprocessor), ("clf", best_clf)])
pipeline.fit(X_train, y_train)

#Avaliação 
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n=== Relatório no conjunto de teste ===")
print(classification_report(y_test, y_pred, target_names=["Presente", "No-Show"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

pathlib.Path("models").mkdir(exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["Presente", "No-Show"], ax=axes[0]
)
axes[0].set_title("Matriz de Confusão")

RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
axes[1].set_title("Curva ROC")
axes[1].plot([0, 1], [0, 1], "k--")

plt.tight_layout()
plt.savefig("models/evaluation.png", dpi=150)
plt.close()
print("Gráficos guardados em models/evaluation.png")

if hasattr(best_clf, "feature_importances_"):
    feat_names = (
        FEATURES_NUM
        + list(pipeline.named_steps["pre"]
               .named_transformers_["cat"]
               .get_feature_names_out(FEATURES_CAT))
        + FEATURES_BIN
    )
    importances = best_clf.feature_importances_
    idx = np.argsort(importances)[::-1][:12]

    plt.figure(figsize=(8, 4))
    plt.barh([feat_names[i] for i in idx[::-1]], importances[idx[::-1]], color="#0D9488")
    plt.xlabel("Importância")
    plt.title("Top Features — No-Show")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png", dpi=150)
    plt.close()
    print("Feature importance guardada em models/feature_importance.png")

with open("models/noshow_model.pkl", "wb") as f:
    pickle.dump({"pipeline": pipeline, "features": ALL_FEATURES,
                 "model_name": best_name}, f)
print("Modelo guardado em models/noshow_model.pkl")
