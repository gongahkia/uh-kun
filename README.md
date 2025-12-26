# uh-kun

An image classification library that categorises images into:
- **Ya-kun**
- **No-kun**
- **Maybe-kun**

It uses **CLIP image embeddings** + **ChromaDB** as a vector database. “Training” means embedding your labelled corpus and storing those embeddings in Chroma; prediction uses nearest-neighbour search and simple vote/score rules.

Label meaning:
- **Ya-kun**: image is *definitely* from Ya Kun Kaya Toast's menu
- **No-kun**: image is *not* from Ya Kun Kaya Toast's menu
- **Maybe-kun**: model is unsure (an abstain/uncertainty output, not a required training class)

## Dataset layout

Put your labelled corpus in a folder like:

```
data/
  train/
    ya-kun/
      img1.jpg
    no-kun/
      img2.png
  val/            # optional
    ya-kun/
    no-kun/
```

Class folder names are case-insensitive and accept a few aliases:
- `ya-kun`, `yakun`, `ya`
- `no-kun`, `nokun`, `no`
- `maybe-kun`, `maybekun`, `maybe`

Note: you can optionally create a `maybe-kun/` folder for experiments, but the intended workflow is to train on **only** `ya-kun/` + `no-kun/` and let the model output **Maybe-kun** when uncertain.

## Install

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Quickstart

### 1) Ingest ("train") your corpus into Chroma

```bash
uh-kun ingest ./data/train --db ./chroma_db --collection yakun
```

### 2) Evaluate (optional)

```bash
uh-kun eval ./data/val --db ./chroma_db --collection yakun --k 5
```

### 3) Predict a single image

```bash
uh-kun predict ./some_image.jpg --db ./chroma_db --collection yakun --k 5
```

If you want the model to abstain more/less often, tune:
- `--maybe-min-score` (default `0.55`)
- `--maybe-margin` (default `0.05`)

## Notes

- This project uses **open-clip-torch** (ViT-B-32 by default) to produce embeddings.
- ChromaDB is used in persistent mode via `--db ./chroma_db`.

## Python API

```python
from uh_kun import UhKun

model = UhKun(db_path="./chroma_db", collection="yakun")
result = model.predict("./some_image.jpg", k=5)
print(result.label, result.scores)
```
