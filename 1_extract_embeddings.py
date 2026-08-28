import os
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector


MODEL_NAME = "microsoft/wavlm-base-plus-sv"

LABELED_DIR = "dataset/labeled"
UNLABELED_DIR = "dataset/unlabeled"

OUTPUT_DIR = "embeddings"
os.mkdir(OUTPUT_DIR)


feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = WavLMForXVector.from_pretrained(MODEL_NAME)
model.eval()


def get_embedding(wav_path):
    audio, sr = sf.read(wav_path)

    if sr != 16000:
        raise ValueError(f"{wav_path}: expected 16 kHz, got {sr}")

    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.inference_mode():
        embedding = model(**inputs).embeddings
        embedding = torch.nn.functional.normalize(
            embedding,
            dim=-1
        )

    return embedding.squeeze(0).numpy()


def process_folder(folder, labeled):
    metadata_path = os.path.join(folder, "metadata.csv")
    df = pd.read_csv(metadata_path)

    embeddings = []
    records = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=os.path.basename(folder)
    ):
        audio_id = row["audio_id"]
        wav_path = os.path.join(folder, f"{audio_id}.wav")

        if not os.path.exists(wav_path):
            print(f"Missing: {wav_path}")
            continue

        try:
            embedding = get_embedding(wav_path)

            embeddings.append(embedding)

            record = {
                "audio_id": audio_id,
                "split": "labeled" if labeled else "unlabeled"
            }

            if labeled:
                record["speaker"] = row["speaker"]
            else:
                record["speaker"] = "Unknown"

            records.append(record)

        except Exception as e:
            print(f"Failed: {audio_id}: {e}")

    return embeddings, records


labeled_embeddings, labeled_records = process_folder(
    LABELED_DIR,
    labeled=True
)

unlabeled_embeddings, unlabeled_records = process_folder(
    UNLABELED_DIR,
    labeled=False
)


all_embeddings = np.stack(
    labeled_embeddings + unlabeled_embeddings
)

index = pd.DataFrame(
    labeled_records + unlabeled_records
)

np.save(
    os.path.join(OUTPUT_DIR, "wavlm_embeddings.npy"),
    all_embeddings
)

index.to_csv(
    os.path.join(OUTPUT_DIR, "embedding_index.csv"),
    index=False
)

print("\nEmbedding matrix:", all_embeddings.shape)
print("Index rows:", len(index))
print("Saved to:", OUTPUT_DIR)
