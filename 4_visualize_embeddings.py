import os, sys, uuid
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import umap


if len(sys.argv) != 2:
    sys.exit("Usage: python 4_visualize_embeddings.py <wavlm|ecapa>")
embedding_name = sys.argv[1]
print(embedding_name)


EMBEDDING_PATH = f"embeddings/{embedding_name}_embeddings.npy"
INDEX_PATH = f"embeddings/{embedding_name}_embedding_index.csv"

OUTPUT_DIR = os.path.join("visualizations", embedding_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)


X = np.load(EMBEDDING_PATH)
meta = pd.read_csv(INDEX_PATH)

assert len(X) == len(meta)

print("Embeddings:", X.shape)
print(meta["speaker"].value_counts())


labeled_mask = meta["speaker"] != "Unknown"

X_labeled = X[labeled_mask]
y_labeled = meta.loc[
    labeled_mask,
    "speaker"
].to_numpy(dtype=str)

speakers = sorted(np.unique(y_labeled))


# colors

palette = px.colors.qualitative.Plotly

speaker_colors = {
    speaker: palette[i % len(palette)]
    for i, speaker in enumerate(speakers)
}

speaker_colors["Unknown"] = "#999999"


def make_3d_figure(coords, title, axis_names):

    fig = go.Figure()

    trace_groups = {
        "labelled": [],
        "unlabelled": []
    }

    # known speakers
    for speaker in speakers:

        mask = meta["speaker"] == speaker

        trace_index = len(fig.data)

        fig.add_trace(
            go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=coords[mask, 2],

                mode="markers",

                name=speaker,

                marker=dict(
                    size=3,
                    opacity=0.65,
                    color=speaker_colors[speaker]
                ),

                text=meta.loc[
                    mask,
                    "audio_id"
                ],

                hovertemplate=(
                    "Speaker: " + speaker +
                    "<br>Audio: %{text}"
                    "<extra></extra>"
                )
            )
        )

        trace_groups["labelled"].append(
            trace_index
        )

    # unknown
    mask = meta["speaker"] == "Unknown"

    trace_index = len(fig.data)

    fig.add_trace(
        go.Scatter3d(
            x=coords[mask, 0],
            y=coords[mask, 1],
            z=coords[mask, 2],

            mode="markers",

            name="Unknown",

            marker=dict(
                size=2.5,
                opacity=0.25,
                color=speaker_colors["Unknown"]
            ),

            text=meta.loc[
                mask,
                "audio_id"
            ],

            hovertemplate=(
                "Speaker: Unknown"
                "<br>Audio: %{text}"
                "<extra></extra>"
            )
        )
    )

    trace_groups["unlabelled"].append(
        trace_index
    )

    fig.update_layout(
        title=title,

        scene=dict(
            xaxis_title=axis_names[0],
            yaxis_title=axis_names[1],
            zaxis_title=axis_names[2]
        ),

        width=1100,
        height=800,

        legend=dict(
            itemsizing="constant"
        )
    )

    return fig, trace_groups


def save_html_with_checkboxes(
    fig,
    trace_groups,
    path
):
    div_id = f"plot_{uuid.uuid4().hex}"

    html = fig.to_html(
        full_html=True,
        include_plotlyjs=True,
        div_id=div_id
    )

    labelled_indices = trace_groups["labelled"]
    unlabelled_indices = trace_groups["unlabelled"]

    controls = f"""
    <div style="
        font-family: Arial, sans-serif;
        margin: 15px;
        font-size: 16px;
    ">
        <label style="margin-right: 20px;">
            <input
                type="checkbox"
                id="show_labelled"
                checked
            >
            Labelled
        </label>

        <label>
            <input
                type="checkbox"
                id="show_unlabelled"
                checked
            >
            Unlabelled
        </label>
    </div>
    """

    script = f"""
    <script>
        const plot = document.getElementById("{div_id}");

        const labelledIndices = {labelled_indices};
        const unlabelledIndices = {unlabelled_indices};

        document
            .getElementById("show_labelled")
            .addEventListener("change", function() {{
                Plotly.restyle(
                    plot,
                    {{visible: this.checked}},
                    labelledIndices
                );
            }});

        document
            .getElementById("show_unlabelled")
            .addEventListener("change", function() {{
                Plotly.restyle(
                    plot,
                    {{visible: this.checked}},
                    unlabelledIndices
                );
            }});
    </script>
    """

    html = html.replace(
        "<body>",
        "<body>" + controls
    )

    html = html.replace(
        "</body>",
        script + "</body>"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(html)


# LDA 3D

print("\nFitting LDA")

lda = LinearDiscriminantAnalysis(
    n_components=3
)

lda.fit(
    X_labeled,
    y_labeled
)

# project both labeled and unknown samples using the projection learned from labeled speakers
X_lda = lda.transform(X)

print(
    "LDA explained variance:",
    lda.explained_variance_ratio_[:3]
)

print(
    "LDA cumulative variance:",
    lda.explained_variance_ratio_[:3].sum()
)


fig, groups = make_3d_figure(
    X_lda,
    f"{embedding_name.upper()} Speaker Embeddings - 3D LDA",
    ("LD1", "LD2", "LD3")
)

save_html_with_checkboxes(
    fig,
    groups,
    os.path.join(
        OUTPUT_DIR,
        "lda_3d.html"
    )
)


# UMAP 3D

print("\nFitting 3D UMAP")

reducer = umap.UMAP(
    n_components=3,
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)

X_umap = reducer.fit_transform(X)


fig, groups = make_3d_figure(
    X_umap,
    f"{embedding_name.upper()} Speaker Embeddings - 3D UMAP",
    ("UMAP 1", "UMAP 2", "UMAP 3")
)

save_html_with_checkboxes(
    fig,
    groups,
    os.path.join(
        OUTPUT_DIR,
        "umap_3d.html"
    )
)


# cosine similarity between speaker centroids

print("\nComputing speaker centroids")

centroids = []

for speaker in speakers:

    mask = meta["speaker"] == speaker

    centroid = X[mask].mean(axis=0)

    # normalize centroid
    centroid = centroid / np.linalg.norm(
        centroid
    )

    centroids.append(centroid)

centroids = np.stack(centroids)


similarity = cosine_similarity(
    centroids
)


similarity_df = pd.DataFrame(
    similarity,
    index=speakers,
    columns=speakers
)

print("\nCentroid cosine similarity:")
print(similarity_df.round(3))


plt.figure(
    figsize=(9, 7)
)

sns.heatmap(
    similarity_df,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    square=True,
    vmin=0,
    vmax=1
)

plt.title(
    f"{embedding_name.upper()} Speaker Centroid Cosine Similarity"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "centroid_cosine_similarity.png"
    ),
    dpi=250
)

plt.close()


# save coordinates

analysis = meta.copy()

analysis["lda_1"] = X_lda[:, 0]
analysis["lda_2"] = X_lda[:, 1]
analysis["lda_3"] = X_lda[:, 2]

analysis["umap_1"] = X_umap[:, 0]
analysis["umap_2"] = X_umap[:, 1]
analysis["umap_3"] = X_umap[:, 2]

analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "embedding_coordinates.csv"
    ),
    index=False
)

print("\nSaved to:", OUTPUT_DIR)
