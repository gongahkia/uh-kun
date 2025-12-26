from __future__ import annotations

from dataclasses import dataclass

from .predictor import PredictConfig, predict_one
from .trainer import EvalConfig, IngestConfig, eval_corpus, ingest_corpus
from .types import Prediction


@dataclass
class UhKun:
    db_path: str
    collection: str = "yakun"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"

    def ingest(self, data_root: str, batch_size: int = 32) -> dict:
        return ingest_corpus(
            IngestConfig(
                data_root=data_root,
                db_path=self.db_path,
                collection=self.collection,
                batch_size=batch_size,
                clip_model=self.clip_model,
                clip_pretrained=self.clip_pretrained,
            )
        )

    def eval(self, data_root: str, k: int = 5, *, maybe_min_score: float = 0.55, maybe_margin: float = 0.05) -> dict:
        return eval_corpus(
            EvalConfig(
                data_root=data_root,
                db_path=self.db_path,
                collection=self.collection,
                k=k,
                maybe_min_score=maybe_min_score,
                maybe_margin=maybe_margin,
                clip_model=self.clip_model,
                clip_pretrained=self.clip_pretrained,
            )
        )

    def predict(
        self,
        image_path: str,
        k: int = 5,
        *,
        maybe_min_score: float = 0.55,
        maybe_margin: float = 0.05,
    ) -> Prediction:
        return predict_one(
            image_path,
            PredictConfig(
                db_path=self.db_path,
                collection=self.collection,
                k=k,
                maybe_min_score=maybe_min_score,
                maybe_margin=maybe_margin,
                clip_model=self.clip_model,
                clip_pretrained=self.clip_pretrained,
            ),
        )
