# Recovering speaker identity from partially labeled Dzongkha speech

We were recording spoken Dzongkha at the NoMind office in Motithang. One afternoon, Ugsy (the bossman) decided to add login credentials for each speaker so that we'd know who the speaker was for a given recording. The issue was, by that time the speakers were already done recording half the dataset, so the speaker labels would only be available for the latter half of the dataset. The majority contributor to the speech recordings was one dude, let's call him "Romeo". I want all of Romeo's recordings in one place (so that I can train a TTS model on just his voice later on). 

I'll first extract speaker embeddings and then use these embeddings for:
- Supervised speaker identification (classification)
- Unsupervised speaker identification (clustering)
- Speaker embedding visualization with PCA and UMAP
- Identifying likely Romeo recordings from the unlabeled clips


## Step 1: Extract speaker embeddings
I used pretrained models to extract a speaker embedding for each recording. I used two models: WavLM Base+ [[1]](#ref-1) and ECAPA-TDNN [[2]](#ref-2). <br>
The `microsoft/wavlm-base-plus-sv` and `speechbrain/spkrec-ecapa-voxceleb` models are used only as feature extractors. <br>

For embeddings extracted from both models, each embedding is L2-normalized, and then all embeddings are saved as an npy file, a multi-dimensional numpy array of shape `(9996, 192)` for ECAPA-TDNN, and `(9996, 512)` for WavLM. <br>
We have 9996 recorded clips. For each recording, ECAPA-TDNN produces an embedding vector of length 192, and WavLM produces an embedding vector of length 512. <br>
An index csv also saved for each npy file, containing the fields: `audio_id`, `split` (labeled|unlabeled), `speaker`.

Extracting embeddings for each recording provides a common representation for every recorded clip. This can then be used for visualization, clustering, and classification. 


## Step 2: Supervised speaker identification (classification)

### A. Binary classification
...

### B. Multi-class classification
...


## Step 3: Unsupervised speaker identification (clustering)
[TO-DO]


## Step 4: Embedding visualization
[TO-DO]


## Step 5: Identifying likely Romeo recordings from the unlabeled clips
[TO-DO]


## Notes
The dataset is not included in this repository (Ugsy would not approve of me open-sourcing the dataset). The embeddings are included though. 


## References

<a id="ref-1"></a>
[1] S. Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing," 2021. https://arxiv.org/abs/2110.13900
<br> Microsoft, "WavLM Base Plus for Speaker Verification". https://huggingface.co/microsoft/wavlm-base-plus-sv

<a id="ref-2"></a>
[2] B. Desplanques, J. Thienpondt, and K. Demuynck, "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification," 2020. https://arxiv.org/abs/2005.07143
<br> SpeechBrain, "ECAPA-TDNN Speaker Recognition Model trained on VoxCeleb." https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
