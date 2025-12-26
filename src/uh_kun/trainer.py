from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .dataset import scan_image_folder
from .embedding import ClipConfig, ClipEmbedder
from .io import load_image
from .types import Label
from .vectordb.chroma import ChromaConfig, ChromaVectorDB


@dataclass(frozen=True)
class IngestConfig:
    data_root: str
    db_path: str
    collection: str = "yakun"
    batch_size: int = 32
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"


def ingest_corpus(cfg: IngestConfig) -> dict:
    items = scan_image_folder(cfg.data_root)
    embedder = ClipEmbedder(ClipConfig(model_name=cfg.clip_model, pretrained=cfg.clip_pretrained))
    vdb = ChromaVectorDB(ChromaConfig(db_path=cfg.db_path, collection=cfg.collection))

    n = 0
    for i in tqdm(range(0, len(items), cfg.batch_size), desc="Ingest"):
        batch = items[i : i + cfg.batch_size]
        images = [load_image(it.path) for it in batch]
        vecs = embedder.embed_pil(images)  # (B, D)

        ids = [it.id for it in batch]
        metadatas = [
            {"label": it.label, "path": str(Path(cfg.data_root) / it.id)} for it in batch
        ]
        vdb.upsert(
            ids=ids,
            embeddings=vecs.astype(np.float32).tolist(),
            metadatas=metadatas,
            documents=[str(it.path) for it in batch],
        )
        n += len(batch)

    return {"ingested": n, "collection_count": vdb.count()}


@dataclass(frozen=True)
class EvalConfig:
    data_root: str
    db_path: str
    collection: str = "yakun"
    k: int = 5
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"


def _vote(labels: list[Label], dists: list[float]) -> dict[Label, float]:
    # Chroma returns distances for cosine space (smaller is closer). Convert to weights.
    scores: dict[Label, float] = {"Ya-kun": 0.0, "No-kun": 0.0, "Maybe-kun": 0.0}
    for lab, dist in zip(labels, dists):
        w = 1.0 / max(dist, 1e-6)
        scores[lab] += float(w)
    return scores


def eval_corpus(cfg: EvalConfig) -> dict:
    items = scan_image_folder(cfg.data_root)
    embedder = ClipEmbedder(ClipConfig(model_name=cfg.clip_model, pretrained=cfg.clip_pretrained))
    vdb = ChromaVectorDB(ChromaConfig(db_path=cfg.db_path, collection=cfg.collection))

    correct = 0
    total = 0

    for it in tqdm(items, desc="Eval"):
        img = load_image(it.path)
        q = embedder.embed_one_pil(img).tolist()
        res = vdb.query(q, k=cfg.k)

        metadatas = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        neigh_labels: list[Label] = [m["label"] for m in metadatas]
        scores = _vote(neigh_labels, distances)
        pred = max(scores, key=scores.get)

        total += 1
        if pred == it.label:
            correct += 1

    acc = (correct / total) if total else 0.0
    return {"accuracy": acc, "correct": correct, "total": total}
