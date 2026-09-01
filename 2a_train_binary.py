import os, sys, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, roc_auc_score
)


if len(sys.argv) != 2:
    sys.exit("Usage: python train_binary.py <wavlm|ecapa>")
embedding_name = sys.argv[1]
print(embedding_name)


X = np.load(f"embeddings/{embedding_name}_embeddings.npy")
meta = pd.read_csv(f"embeddings/{embedding_name}_embedding_index.csv")

# keep only labeled samples
mask = meta["speaker"] != "Unknown"
X = X[mask]
meta = meta[mask].reset_index(drop=True)

# romeo = 1, non-romeo = 0
y = (meta["speaker"] == "Romeo").astype(int).values

print("samples:", len(y))
print("romeo:", y.sum())
print("non-romeo:", (y == 0).sum())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("train samples:", len(X_train))
print("test samples:", len(X_test))


def train_and_evaluate(model, model_name, class_names=("Non-Romeo", "Romeo")):
    
    output_dir = os.path.join("trained_models", "binary", embedding_name, model_name)
    os.makedirs(output_dir, exist_ok=True)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    roc_auc = (
        roc_auc_score(y_test, y_score)
        if y_score is not None
        else None
    )

    report = classification_report(
        y_test, y_pred,
        target_names=class_names
    )

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{model_name}")
    print("Accuracy:", accuracy)
    print("Balanced accuracy:", balanced_accuracy)

    if roc_auc is not None:
        print("ROC-AUC:", roc_auc)

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(report)

    # save report
    report_path = os.path.join(
        output_dir,
        "classification_report.txt"
    )
    
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy:.6f}\n")
        f.write(f"Balanced accuracy: {balanced_accuracy:.6f}\n")

        if roc_auc is not None:
            f.write(f"ROC-AUC: {roc_auc:.6f}\n")

        f.write("\nConfusion matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification report:\n")
        f.write(report)

    # save confusion matrix
    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(model_name)
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "confusion_matrix.png"), dpi=200
    )

    plt.close()

    # save trained model
    joblib.dump(
        model,
        os.path.join(output_dir, "model.joblib")
    )


models = {
    "logistic_regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),

    "linear_svm": SVC(
        kernel="linear",
        class_weight="balanced"
    ),

    "rbf_svm": SVC(
        kernel="rbf",
        class_weight="balanced"
    ),

    "knn": KNeighborsClassifier(
        n_neighbors=5,
        metric="cosine",
        weights="distance"
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
}


for model_name, model in models.items():
    train_and_evaluate(model, model_name)
