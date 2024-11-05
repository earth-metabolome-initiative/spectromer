"""Script to visualize the matchms results for the NPC dataset."""

from typing import List, Dict, Union, cast
import os
from multiprocessing import cpu_count
import numpy as np
import matplotlib.pyplot as plt
import compress_json
from tqdm.auto import tqdm
from matchms.importing import load_from_mgf
from sklearn.decomposition import PCA  # type: ignore
from openTSNE import TSNE
from matplotlib.colors import TABLEAU_COLORS as TABLEAU_COLORS_DICT


TABLEAU_COLORS: List[str] = list(TABLEAU_COLORS_DICT.keys())


def visualize_feature() -> None:
    """Visualize a feature."""

    if not os.path.exists("matchms_smiles.json.gz"):
        all_smiles: List[str] = []

        for spectra in tqdm(
            load_from_mgf("../datasets/GNPS/matchms.mgf"),
            desc="Retrieving SMILES",
            unit="spectrum",
            leave=False,
            dynamic_ncols=True,
        ):
            all_smiles.append(spectra.get("smiles"))
        
        compress_json.dump(all_smiles, "matchms_smiles.json.gz")
    else:
        all_smiles = compress_json.load("matchms_smiles.json.gz")

    labels: List[Dict[str, Union[str, List[str]]]] = compress_json.load(
        "classified_matchms.json.gz"
    )
    smiles_to_labels: Dict[str, Dict[str, Union[str, List[str]]]] = {
        cast(str, label["smiles"]): label for label in labels
    }

    pathway_ids: Dict[str, int] = {}
    pathways: np.ndarray = np.zeros((len(all_smiles)), dtype=int)

    for i, smiles in enumerate(all_smiles):
        smiles_labels = smiles_to_labels[smiles]
        if "pathway_results" not in smiles_labels:
            smiles_labels["pathway_results"] = ["Unknown"]
        for pathway in smiles_labels["pathway_results"]:
            if pathway not in pathway_ids:
                pathway_ids[pathway] = len(pathway_ids)
            pathways[i] = pathway_ids[pathway]

    embedding: np.ndarray = np.load("../matchms.npy")

    if embedding.shape[1] > 50:
        embedding = PCA(n_components=50).fit_transform(embedding)

    pca_embedding = PCA(n_components=2).fit_transform(embedding)

    tsne = TSNE(
        n_components=2,
        n_jobs=cpu_count(),
        verbose=True,
        early_exaggeration_iter=250,
        n_iter=1000,
    )

    tsne_embedding: np.ndarray = tsne.fit(embedding)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=300)

    colors = [TABLEAU_COLORS[i] for i in pathways]

    for i, (embedding, title) in enumerate(
        [(pca_embedding, "PCA"), (tsne_embedding, "t-SNE")]
    ):
        axes[i].scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=colors,
            s=10,
            alpha=0.5,
        )
        axes[i].set_title(title)

    axes[0].legend(
        handles=[
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=TABLEAU_COLORS[i]
            )
            for i in range(len(pathway_ids))
        ],
        labels=list(pathway_ids.keys()),
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    fig.tight_layout()
    fig.savefig("embedding.png")


if __name__ == "__main__":
    visualize_feature()
