"""임베딩 기반 외부 기억: 루프가 매 실행마다 읽어 가는, 모델 밖에 사는 지식.

04(메모리·RAG), 10(일일 트리아지), 11(중복 이슈 탐지), 15(기술부채 탐지)가 이
모듈을 쓴다. 코퍼스를 한 번 임베딩해 두고, 질의와 코사인 유사도가 높은 상위 k개를
돌려주는 것이 전부다.

백엔드는 두 가지이고, 인자를 비워 두면 실행 환경을 보고 자동으로 정해진다.

    mem = EmbeddingMemory()        # 자동: GPU 있으면 bge-large, 없으면 API 또는 bge-small
    mem = EmbeddingMemory(backend="openai")   # 강제 지정

    local  : sentence-transformers (BAAI/bge-large-en-v1.5 / bge-small-en-v1.5)
    openai : text-embedding-3-small / -large

주의: 두 백엔드는 쿼리 처리 규약이 다르다. BGE 계열은 비대칭 검색(짧은 질의로 긴
문서를 찾는 상황)에서 질의 앞에 "Represent this sentence for..." 지시문을 붙여야
성능이 나오도록 학습됐다. OpenAI 임베딩에는 그런 규약이 없어서 지시문을 붙이면
오히려 노이즈가 된다. runtime 이 백엔드에 맞는 지시문을 넣어 준다.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import runtime


class EmbeddingMemory:
    """코퍼스를 임베딩해 두고 유사도 상위 k개를 검색한다."""

    def __init__(
        self,
        backend: str | None = None,
        model_name: str | None = None,
        device: str | None = None,
        query_instruction: str | None = None,
        cache_dir: str | None = None,
    ):
        # 비워 둔 인자는 실행 환경 판별 결과로 채운다.
        cfg = runtime.resolve_embed(force=backend) if backend else runtime.get_runtime().embed

        self.backend = cfg.provider
        self.model_name = model_name or cfg.model
        self.device = device or cfg.device
        self.query_instruction = cfg.query_instruction if query_instruction is None else query_instruction
        self.resolve_reason = cfg.reason

        self.docs: list[str] = []
        self.meta: list = []
        self.emb: np.ndarray | None = None

        # API 임베딩은 호출마다 과금되므로 디스크 캐시가 필수다.
        # 11번은 코퍼스 4,000건을 임베딩한다. 캐시가 없으면 재실행할 때마다
        # 같은 비용이 그대로 다시 발생한다.
        self.cache_dir = Path(cache_dir) if cache_dir else runtime.project_root() / ".cache" / "emb"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model = None      # sentence-transformers 모델 (지연 로딩)
        self._client = None     # OpenAI 클라이언트 (지연 로딩)
        self._cfg = cfg

    # ── 백엔드 지연 초기화 ──────────────────────────────────────────
    def _ensure_local(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _ensure_openai(self):
        if self._client is None:
            from openai import OpenAI
            kwargs = {"api_key": self._cfg.api_key or os.environ.get("OPENAI_API_KEY")}
            if self._cfg.base_url:
                kwargs["base_url"] = self._cfg.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    # ── 캐시 ────────────────────────────────────────────────────────
    def _cache_key(self, text: str) -> str:
        """모델이 다르면 벡터도 다르므로 모델명을 키에 포함한다."""
        h = hashlib.sha1(f"{self.backend}|{self.model_name}|{text}".encode()).hexdigest()
        return h

    def _cache_load(self, keys: list[str]) -> dict[str, np.ndarray]:
        out = {}
        for k in keys:
            p = self.cache_dir / f"{k}.npy"
            if p.exists():
                try:
                    out[k] = np.load(p)
                except Exception:
                    pass
        return out

    def _cache_save(self, key: str, vec: np.ndarray) -> None:
        try:
            np.save(self.cache_dir / f"{key}.npy", vec)
        except Exception:
            pass   # 캐시는 실패해도 본 작업을 막지 않는다

    # ── 임베딩 ──────────────────────────────────────────────────────
    def _embed_local(self, texts: list[str], batch_size: int) -> np.ndarray:
        model = self._ensure_local()
        v = model.encode(texts, normalize_embeddings=True, batch_size=batch_size,
                         show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)

    def _embed_openai(self, texts: list[str], batch_size: int) -> np.ndarray:
        client = self._ensure_openai()
        keys = [self._cache_key(t) for t in texts]
        cached = self._cache_load(keys)

        # 캐시에 없는 것만 API 로 보낸다
        todo = [(i, t, k) for i, (t, k) in enumerate(zip(texts, keys)) if k not in cached]
        for start in range(0, len(todo), batch_size):
            chunk = todo[start:start + batch_size]
            resp = client.embeddings.create(
                model=self.model_name,
                input=[t for _, t, _ in chunk],
            )
            for (i, _t, k), item in zip(chunk, resp.data):
                vec = np.asarray(item.embedding, dtype=np.float32)
                # 코사인 유사도를 내적으로 계산하려면 정규화가 필요하다.
                # sentence-transformers 의 normalize_embeddings=True 와 맞추는 것이다.
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                cached[k] = vec
                self._cache_save(k, vec)

        return np.stack([cached[k] for k in keys]).astype(np.float32)

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """텍스트 목록을 정규화된 벡터 행렬로. 백엔드와 무관하게 같은 모양이다."""
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if self.backend == "openai":
            return self._embed_openai(texts, batch_size=min(batch_size, 128))
        return self._embed_local(texts, batch_size)

    # ── 코퍼스 관리 (원본과 동일한 인터페이스) ──────────────────────
    def add(self, docs: list[str], meta: list | None = None) -> None:
        self.docs.extend(docs)
        self.meta.extend(meta if meta is not None else [None] * len(docs))

    def index(self, batch_size: int = 64) -> None:
        self.emb = self.embed_texts(self.docs, batch_size=batch_size)

    def _encode_query(self, query: str) -> np.ndarray:
        # BGE 는 지시문을 붙이고, OpenAI 는 붙이지 않는다 (모듈 docstring 참고).
        q = self.query_instruction + query
        v = self.embed_texts([q], batch_size=1)[0]
        return np.asarray(v, dtype=np.float32)

    def retrieve(self, query: str, k: int = 3, exclude_idx: int | None = None) -> list[dict]:
        """유사도 상위 k개를 {index, score, doc, meta} 형태로 돌려준다."""
        if self.emb is None:
            self.index()
        sims = self.emb @ self._encode_query(query)   # 정규화돼 있으므로 내적 = 코사인
        order = np.argsort(-sims)
        out = []
        for i in order:
            if exclude_idx is not None and int(i) == exclude_idx:
                continue
            out.append({"index": int(i), "score": float(sims[i]),
                        "doc": self.docs[i], "meta": self.meta[i]})
            if len(out) >= k:
                break
        return out

    def describe(self) -> dict:
        """어떤 백엔드로 동작 중인지. 노트북에서 찍어 보기 좋다."""
        n_cached = len(list(self.cache_dir.glob("*.npy"))) if self.cache_dir.exists() else 0
        return {
            "backend": self.backend,
            "model": self.model_name,
            "device": self.device or "(원격)",
            "query_instruction": bool(self.query_instruction),
            "docs": len(self.docs),
            "cached_vectors": n_cached,
            "resolved_by": self.resolve_reason,
        }
