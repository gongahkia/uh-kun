from __future__ import annotations

from pathlib import Path

import typer
from rich import print
from rich.table import Table

from .api import UhKun
from .trainer import EvalConfig, IngestConfig, eval_corpus, ingest_corpus
from .predictor import PredictConfig, predict_one

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def ingest(
    data_root: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    db: Path = typer.Option(..., "--db", help="ChromaDB persistent directory"),
    collection: str = typer.Option("yakun", "--collection"),
    batch_size: int = typer.Option(32, "--batch-size"),
    k: int = typer.Option(5, "--k", help="Unused for ingest; kept for symmetry"),
):
    """Embed and store a labelled corpus into ChromaDB ("training")."""
    _ = k
    res = ingest_corpus(
        IngestConfig(
            data_root=str(data_root),
            db_path=str(db),
            collection=collection,
            batch_size=batch_size,
        )
    )
    print(res)


@app.command()
def eval(
    data_root: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    db: Path = typer.Option(..., "--db", help="ChromaDB persistent directory"),
    collection: str = typer.Option("yakun", "--collection"),
    k: int = typer.Option(5, "--k"),
):
    """Evaluate accuracy against a labelled validation folder."""
    res = eval_corpus(EvalConfig(data_root=str(data_root), db_path=str(db), collection=collection, k=k))
    print(res)


@app.command()
def predict(
    image: Path = typer.Argument(..., exists=True, dir_okay=False, file_okay=True),
    db: Path = typer.Option(..., "--db", help="ChromaDB persistent directory"),
    collection: str = typer.Option("yakun", "--collection"),
    k: int = typer.Option(5, "--k"),
):
    """Predict Ya-kun / No-kun / Maybe-kun for a single image."""
    pred = predict_one(str(image), PredictConfig(db_path=str(db), collection=collection, k=k))

    print(f"[bold]Prediction:[/bold] {pred.label}")
    t = Table(title="Scores")
    t.add_column("Label")
    t.add_column("Score", justify="right")
    for lab, score in pred.scores.items():
        t.add_row(lab, f"{score:.4f}")
    print(t)

    t2 = Table(title=f"Nearest Neighbors (k={k})")
    t2.add_column("ID")
    t2.add_column("Label")
    t2.add_column("Distance", justify="right")
    for _id, lab, dist in pred.neighbors:
        t2.add_row(_id, lab, f"{dist:.4f}")
    print(t2)


@app.command()
def api_example(
    db: Path = typer.Option(..., "--db"),
    collection: str = typer.Option("yakun", "--collection"),
):
    """Tiny sanity check that constructs the Python API."""
    model = UhKun(db_path=str(db), collection=collection)
    print(model)
