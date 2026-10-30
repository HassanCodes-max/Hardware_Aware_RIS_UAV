"""
Vision-aided beam prediction on DeepSense 6G Scenario 23.

The base station's camera (960 x 540 RGB, one frame per beam sweep) is
encoded by a frozen ResNet-18 pretrained on ImageNet; no layer is trained
on our data. The 512-d global-pooled embedding of every frame is cached
once (``--extract``). Two uses:

1. camera only: a classifier maps the frame at t to the best beam at
   t + h (h = 0, 1, 3), with no power measurement at all;
2. camera as side information: ``SCENARIO23_FEATURES=camera`` (or
   ``gps+camera``) appends a 16-component PCA of the embedding to every
   step of beam_prediction.py and beam_sequence_model.py.

PCA is fitted on the embeddings of all frames without labels; the
classifiers that see labels are fitted inside GroupKFold over seq_index,
as everywhere else. Run with the full Scenario 23 download in
public_data/scenario23_dev (unit1/camera_data).
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from beam_prediction import BEAMS, DATA_DIR, FOLDS, RESULTS_DIR, SEED, link_metrics, load_scenario23

EMBEDDING_FILE = DATA_DIR / "camera_resnet18_embeddings.npz"
IMAGE_SIZE = (224, 398)          # keeps the 16:9 frame; the network pools globally
PCA_COMPONENTS = 16
CAMERA_HORIZONS = (0, 1, 3)
BATCH = 64


def extract_embeddings() -> None:
    """Encode every camera frame with a frozen ImageNet ResNet-18 and cache it."""

    import torch
    from PIL import Image
    from torchvision.models import ResNet18_Weights, resnet18
    from torchvision.transforms import functional as F

    torch.set_num_threads(8)
    manifest = pd.read_csv(DATA_DIR / "scenario23.csv")
    paths = [DATA_DIR / str(p).replace("./", "") for p in manifest["unit1_rgb"]]

    network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    network.fc = torch.nn.Identity()
    network.eval()
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def load(path):
        image = F.resize(F.to_tensor(Image.open(path).convert("RGB")), list(IMAGE_SIZE),
                         antialias=True)
        return (image - mean) / std

    embeddings = np.zeros((len(paths), 512), dtype=np.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(paths), BATCH):
            batch = torch.stack([load(p) for p in paths[start:start + BATCH]])
            embeddings[start:start + BATCH] = network(batch).numpy()
            if start % (BATCH * 20) == 0:
                print(f"  {start}/{len(paths)} frames, {time.perf_counter() - started:.0f} s")

        # Batch-1 latency of the encoder alone, one thread, as for the other models.
        torch.set_num_threads(1)
        frame = load(paths[0])[None]
        for _ in range(5):
            network(frame)
        timings = []
        for _ in range(50):
            tic = time.perf_counter()
            network(frame)
            timings.append((time.perf_counter() - tic) * 1e3)

    np.savez_compressed(EMBEDDING_FILE, embeddings=embeddings,
                        index=manifest["index"].to_numpy(),
                        encoder_ms_median=np.median(timings))
    print(f"Saved {EMBEDDING_FILE} ({len(paths)} frames, "
          f"encoder {np.median(timings):.1f} ms/frame on one CPU thread)")


def load_embeddings() -> tuple[np.ndarray, float]:
    if not EMBEDDING_FILE.exists():
        raise FileNotFoundError(f"{EMBEDDING_FILE} missing: run camera_beam_prediction.py --extract")
    cache = np.load(EMBEDDING_FILE)
    return cache["embeddings"].astype(float), float(cache["encoder_ms_median"])


def camera_components(components: int = PCA_COMPONENTS) -> np.ndarray:
    """Label-free PCA of the frame embeddings, for use as per-step side information."""

    embeddings, _ = load_embeddings()
    return PCA(n_components=components, random_state=SEED).fit_transform(
        StandardScaler().fit_transform(embeddings))


def classifiers() -> dict:
    return {
        "camera logistic": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)),
        "camera MLP": make_pipeline(
            StandardScaler(), MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2,
                                            early_stopping=True, max_iter=300,
                                            random_state=SEED)),
    }


def camera_only(embeddings: np.ndarray) -> list[dict]:
    """Frame at t -> best beam at t + h, with no power measurement."""

    power, labels, sequences, _, _ = load_scenario23()
    rows = []
    for horizon in CAMERA_HORIZONS:
        index = np.arange(len(labels) - horizon)
        index = index[sequences[index] == sequences[index + horizon]]
        features = embeddings[index]
        target = labels[index + horizon]
        target_power = power[index + horizon]
        groups = sequences[index]
        print(f"\nCamera only, t+{horizon}: {len(index)} samples")

        for fold, (train, test) in enumerate(GroupKFold(n_splits=FOLDS).split(features, target, groups)):
            for name, model in classifiers().items():
                model.fit(features[train], target[train])
                scores = np.zeros((len(test), BEAMS))
                scores[:, model.classes_] = model.predict_proba(features[test])
                rows.append({"horizon": horizon, "model": name, "fold": fold,
                             **link_metrics(scores, target[test], target_power[test])})
                print(f"  fold {fold} {name}: within-noise {rows[-1]['within_noise_acc']:.3f}")

            # Reference: the training folds' most frequent beam (no input at all).
            prior = np.bincount(target[train], minlength=BEAMS).astype(float)
            rows.append({"horizon": horizon, "model": "most frequent beam", "fold": fold,
                         **link_metrics(np.tile(prior, (len(test), 1)), target[test],
                                        target_power[test])})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--extract", action="store_true",
                        help="encode all camera frames first (needs torchvision)")
    args = parser.parse_args()

    if args.extract or not EMBEDDING_FILE.exists():
        extract_embeddings()
    embeddings, encoder_ms = load_embeddings()

    table = pd.DataFrame(camera_only(embeddings))
    summary = table.drop(columns="fold").groupby(["horizon", "model"]).mean()
    pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED).fit(
        StandardScaler().fit_transform(embeddings))

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "camera_beam_prediction_folds.csv", index=False)
    (RESULTS_DIR / "camera_beam_prediction.json").write_text(json.dumps({
        "split": f"GroupKFold({FOLDS}) over seq_index",
        "encoder": "ResNet-18, ImageNet weights, frozen, input 224x398",
        "encoder_ms_median_one_thread": round(encoder_ms, 2),
        "pca_components": PCA_COMPONENTS,
        "pca_explained_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
        "summary": [
            {"horizon": h, "model": m, **values}
            for (h, m), values in summary.round(4).to_dict(orient="index").items()
        ],
    }, indent=2))

    columns = ["top1_acc", "within_noise_acc", "balanced_within_noise_acc", "top3_acc", "macro_f1"]
    print()
    print(summary[columns].round(3).to_string())


if __name__ == "__main__":
    main()
