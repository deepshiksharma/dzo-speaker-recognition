import os, shutil, joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC


EMBEDDING_PATH = "embeddings/ecapa_embeddings.npy"
INDEX_PATH = "embeddings/ecapa_embedding_index.csv"

UNLABELED_WAV_DIR = "dataset/unlabeled"

OUTPUT_DIR = "classified"
ROMEO_DIR = os.path.join(OUTPUT_DIR, "romeo")
NON_ROMEO_DIR = os.path.join(OUTPUT_DIR, "non_romeo")

MODEL_DIR = "trained_models/final"
MODEL_PATH = os.path.join(MODEL_DIR, "ecapa_rbf_svm.joblib")

PREDICTIONS_PATH = os.path.join(
    OUTPUT_DIR,
    "predictions.csv"
)


os.makedirs(ROMEO_DIR, exist_ok=True)
os.makedirs(NON_ROMEO_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# Load ECAPA embeddings and metadata
X = np.load(EMBEDDING_PATH)
meta = pd.read_csv(INDEX_PATH)

assert len(X) == len(meta)


# Split labeled and unknown embeddings
labeled_mask = meta["speaker"] != "Unknown"
unknown_mask = meta["speaker"] == "Unknown"

X_labeled = X[labeled_mask]
X_unknown = X[unknown_mask]

labeled_meta = meta[labeled_mask].reset_index(drop=True)
unknown_meta = meta[unknown_mask].reset_index(drop=True)


# Romeo = 1, Non-Romeo = 0
y_labeled = (
    labeled_meta["speaker"] == "Romeo"
).astype(int).to_numpy()


print("Labeled samples:", len(X_labeled))
print("Romeo:", y_labeled.sum())
print("Non-Romeo:", (y_labeled == 0).sum())
print("Unknown samples:", len(X_unknown))


# Train final model on ALL labeled data
model = SVC(
    kernel="rbf",
    class_weight="balanced"
)

print("\nTraining final RBF SVM...")

model.fit(
    X_labeled,
    y_labeled
)

joblib.dump(
    model,
    MODEL_PATH
)

print("Saved model:", MODEL_PATH)


# Predict all unknown recordings
predictions = model.predict(X_unknown)

# Positive = more Romeo-like
# Negative = more Non-Romeo-like
decision_scores = model.decision_function(
    X_unknown
)


results = unknown_meta[
    ["audio_id"]
].copy()

results["prediction"] = np.where(
    predictions == 1,
    "Romeo",
    "Non-Romeo"
)

results["decision_score"] = decision_scores


# Sort by Romeo confidence
results = results.sort_values(
    "decision_score",
    ascending=False
).reset_index(drop=True)


results.to_csv(
    PREDICTIONS_PATH,
    index=False
)


print("\nPredictions:")
print(results["prediction"].value_counts())

print("\nSaved predictions:")
print(PREDICTIONS_PATH)


# Copy WAV files
missing = 0

for _, row in results.iterrows():

    audio_id = row["audio_id"]
    prediction = row["prediction"]

    src = os.path.join(
        UNLABELED_WAV_DIR,
        f"{audio_id}.wav"
    )

    if prediction == "Romeo":
        dst = os.path.join(
            ROMEO_DIR,
            f"{audio_id}.wav"
        )
    else:
        dst = os.path.join(
            NON_ROMEO_DIR,
            f"{audio_id}.wav"
        )

    if not os.path.exists(src):
        print("Missing:", src)
        missing += 1
        continue

    # Skip files already copied on a previous run
    if not os.path.exists(dst):
        shutil.copy2(
            src,
            dst
        )


print("\nDone.")
print("Romeo WAVs:", ROMEO_DIR)
print("Non-Romeo WAVs:", NON_ROMEO_DIR)
print("Missing WAVs:", missing)
