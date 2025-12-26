from __future__ import annotations

from dataclasses import dataclass

from .embedding import ClipConfig, ClipEmbedder
from .io import load_image
from .types import Label, Prediction
from .vectordb.chroma import ChromaConfig, ChromaVectorDB


@dataclass(frozen=True)
class PredictConfig:
    db_path: str
    collection: str = "yakun"
    k: int = 5
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"


def _vote(labels: list[Label], dists: list[float]) -> dict[Label, float]:
    scores: dict[Label, float] = {"Ya-kun": 0.0, "No-kun": 0.0, "Maybe-kun": 0.0}
    for lab, dist in zip(labels, dists):
        w = 1.0 / max(dist, 1e-6)
        scores[lab] += float(w)
    # Normalize to sum=1 for readability
    s = sum(scores.values())
    if s > 0:
        for k in list(scores.keys()):
            scores[k] = scores[k] / s
    return scores


def predict_one(image_path: str, cfg: PredictConfig) -> Prediction:
    embedder = ClipEmbedder(ClipConfig(model_name=cfg.clip_model, pretrained=cfg.clip_pretrained))
    vdb = ChromaVectorDB(ChromaConfig(db_path=cfg.db_path, collection=cfg.collection))

    img = load_image(image_path)
    q = embedder.embed_one_pil(img).tolist()
    res = vdb.query(q, k=cfg.k)

    ids = (res.get("ids") or [[]])[0]
    metadatas = (res.get("metadatas") or [[]])[0]
    distances = (res.get("distances") or [[]])[0]

    neigh: list[tuple[str, Label, float]] = []
    neigh_labels: list[Label] = []
    for _id, md, dist in zip(ids, metadatas, distances):
        lab: Label = md["label"]
        neigh.append((_id, lab, float(dist)))
        neigh_labels.append(lab)

    scores = _vote(neigh_labels, distances)
    label = max(scores, key=scores.get)
    return Prediction(label=label, scores=scores, neighbors=neigh)
