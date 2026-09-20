"""Audita si las señales NVML separan compute-bound de memory-bound.

La figura es diagnóstica: PCA conserva varianza global y t-SNE vecindarios
locales. Ninguna de las dos proyecciones se usa para entrenar ni para reportar
la métrica del clasificador. Las medidas se calculan en el espacio original
estandarizado y se contrastan con vecinos de una familia no vista.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from fase2_clasificador.analysis.evaluate_historical_gpu_runs import add_physical_features, feature_variants


def load_frame(path: Path) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    frame = frame.loc[
        frame["training_eligible"].astype(bool)
        & frame["kernel_family"].ne("minibude_cuda_bm1")
    ].copy()
    frame = add_physical_features(frame)
    features = feature_variants()["median_iqr"]
    frame = frame.dropna(subset=features + ["phase_label_train", "kernel_family"]).reset_index(drop=True)
    X = StandardScaler().fit_transform(frame[features].to_numpy(dtype=float))
    # True representa memory_bound; se conserva explícito en las salidas.
    y = frame["phase_label_train"].eq("memory_bound").to_numpy()
    return frame, features, X, y


def leave_family_out_knn(X: np.ndarray, y: np.ndarray, families: np.ndarray, n_neighbors: int = 3) -> np.ndarray:
    """Predicciones OOF de un kNN que jamás ve ejemplos de la familia evaluada."""
    prediction = np.empty(len(y), dtype=bool)
    for family in np.unique(families):
        train = families != family
        test = ~train
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
        model.fit(X[train], y[train])
        prediction[test] = model.predict(X[test])
    return prediction


def plot_projection(ax, coordinates: np.ndarray, frame: pd.DataFrame, title: str) -> None:
    colors = np.where(frame["phase_label_train"].eq("memory_bound"), "#1f77b4", "#d62728")
    ax.scatter(coordinates[:, 0], coordinates[:, 1], c=colors, alpha=0.76, s=24, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.grid(alpha=0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("tmp/historical_gpu_merged_20260917.csv"))
    parser.add_argument("--plot", type=Path, default=Path("tmp/historical_gpu_separability_20260918.png"))
    parser.add_argument("--report", type=Path, default=Path("tmp/historical_gpu_separability_20260918.json"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    frame, features, X, y = load_frame(args.input)
    families = frame["kernel_family"].to_numpy()
    pred = leave_family_out_knn(X, y, families)
    pca = PCA(n_components=2, random_state=args.seed).fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=args.seed).fit_transform(X)

    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    plot_projection(axes[0], pca, frame, "PCA: estructura global")
    plot_projection(axes[1], tsne, frame, "t-SNE: vecindarios locales")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="#d62728", label="compute-bound"),
        plt.Line2D([], [], marker="o", linestyle="", color="#1f77b4", label="memory-bound"),
    ]
    axes[1].legend(handles=handles, loc="best")
    fig.suptitle("Señales NVML estáticas: clases y solapamiento entre familias", fontsize=14)
    fig.savefig(args.plot, dpi=180)
    plt.close(fig)

    report = {
        "dataset": str(args.input),
        "features": features,
        "n_runs": int(len(frame)),
        "n_families": int(frame["kernel_family"].nunique()),
        "class_counts": {
            "compute_bound": int((~y).sum()),
            "memory_bound": int(y.sum()),
        },
        "silhouette_original_space": {
            "by_class": float(silhouette_score(X, y)),
            "by_family": float(silhouette_score(X, families)),
        },
        "leave_family_out_3nn": {
            "accuracy": float(accuracy_score(y, pred)),
            "f1_macro_pooled": float(f1_score(y, pred, labels=[False, True], average="macro", zero_division=0)),
            "confusion_matrix_rows_compute_memory": confusion_matrix(y, pred, labels=[False, True]).tolist(),
        },
        "interpretation": {
            "projection_warning": "t-SNE is only a local-neighborhood visualization; it is not evidence of generalization.",
            "evaluation_warning": "kNN predictions exclude every sample from the held-out kernel family.",
        },
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
