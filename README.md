# Recovering speaker identity from partially labeled Dzongkha speech

We were recording spoken Dzongkha at the NoMind office in Motithang. One afternoon, Ugsy (the bossman) decided to add login credentials for each speaker so that we'd know who the speaker was for a given recording. The issue was, by that time the speakers were already done recording half the dataset, so the speaker labels would only be available for the latter half of the dataset. The majority contributor to the speech recordings was one dude, let's call him "Romeo". I want all of Romeo's recordings in one place (so that I can train a TTS model on just his voice later on). 

I'll first extract speaker embeddings and then use these embeddings for:
- Speaker embedding visualization with PCA and UMAP
- Unsupervised speaker identification (clustering)
- Supervised speaker identification (classification)
- Romeo vs. Non-Romeo (binary classification)
- Identifying likely Romeo recordings from the unlabeled clips


## Phase 1: Extracting embeddings
I used the pretrained WavLM Base+ speaker verification model [[1]](#ref-1) to extract a speaker embedding for each recording.

The `microsoft/wavlm-base-plus-sv` model is used only as a feature extractor. Each embedding is L2-normalized, and then all embeddings are saved in a single npy file as a multi-dimensional numpy array of shape (9996, 512). Each embedding is of length 512, and we have 9996 recorded clips. <br>
An index csv also saved, containing the fields: `audio_id`, `split` (labeled|unlabeled), `speaker`.

Extracting embeddings for each recording provides a common representation for every recorded clip. This can then be used for visualization, clustering, and classification. 


## Phase 2: Embedding visualization
[TO-DO]


## Phase 3: Unsupervised speaker identification (clustering)
[TO-DO]


## Phase 4: Supervised speaker identification (classification)
[TO-DO]


## Phase 5: Romeo vs Non-Romeo (binary classification)
[TO-DO]


## Phase 6: Identifying likely Romeo recordings from the unlabeled clips
[TO-DO]


## Notes
The dataset is not included in this repository (Ugsy would not approve of me open-sourcing the dataset). The embeddings are included though. 


## References

<a id="ref-1"></a>
[1] Microsoft, "WavLM Base Plus for Speaker Verification". https://huggingface.co/microsoft/wavlm-base-plus-sv
