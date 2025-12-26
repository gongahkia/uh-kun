from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import chromadb


@dataclass(frozen=True)
class ChromaConfig:
    db_path: str
    collection: str = "yakun"


class ChromaVectorDB:
    def __init__(self, config: ChromaConfig):
        self.config = config
        self._client = chromadb.PersistentClient(path=config.db_path)
        self._collection = self._client.get_or_create_collection(
            name=config.collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        documents: list[str] | None = None,
    ) -> None:
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    def query(self, embedding: list[float], k: int) -> dict:
        return self._collection.query(query_embeddings=[embedding], n_results=k)

    def count(self) -> int:
        return self._collection.count()

    def iter_ids(self, batch: int = 1000) -> Iterable[str]:
        # Chroma doesn't provide a stable streaming iterator in all versions.
        # We'll pull ids via get() in pages.
        offset = 0
        while True:
            got = self._collection.get(limit=batch, offset=offset, include=[])
            ids = got.get("ids") or []
            if not ids:
                return
            for _id in ids:
                yield _id
            offset += len(ids)

    def delete_ids(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)
