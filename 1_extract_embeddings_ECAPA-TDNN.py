import os
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from tqdm import tqdm
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


META_PATH = "embeddings/embedding_index.csv"
meta = pd.read_csv(META_PATH)

OUTPUT_DIR = "embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)


classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/spkrec-ecapa-voxceleb",
    local_strategy=LocalStrategy.COPY
)


embeddings = []
records = []


for _, row in tqdm(meta.iterrows(), total=len(meta)):
    folder = (
        "dataset/labeled"
        if row["split"] == "labeled"
        else "dataset/unlabeled"
    )

    wav_path = os.path.join(
        folder,
        f"{row['audio_id']}.wav"
    )

    try:
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        signal = torch.from_numpy(audio).unsqueeze(0)

        if sr != 16000:
            raise ValueError(
                f"Expected 16 kHz, got {sr}"
            )

        with torch.inference_mode():
            embedding = classifier.encode_batch(signal)

        embedding = embedding.squeeze()

        embedding = torch.nn.functional.normalize(
            embedding,
            dim=0
        )

        embeddings.append(
            embedding.cpu().numpy()
        )

        records.append({
            "audio_id": row["audio_id"],
            "split": row["split"],
            "speaker": row["speaker"],
        })

    except Exception as e:
        print(
            f"Failed {row['audio_id']}: {e}"
        )


X = np.stack(embeddings)
index = pd.DataFrame(records)

np.save(
    os.path.join(OUTPUT_DIR, "ecapa_embeddings.npy"),
    X
)

index.to_csv(
    os.path.join(OUTPUT_DIR, "ecapa_embedding_index.csv"),
    index=False
)

print("\necapa embedding matrix:", X.shape)
print("index rows:", len(index))
print("saved to:", OUTPUT_DIR)
