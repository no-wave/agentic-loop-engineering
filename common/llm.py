"""LLM 클라이언트: 로컬 vLLM 과 OpenAI API 를 같은 인터페이스로 감싼다.

모든 노트북이 이 모듈을 통해서만 모델과 대화한다. 그래야 토큰 사용량·지연시간·
샘플 수가 모든 실험에서 동일한 방식으로 계측되고, 08번(예산·관측)의 비용/품질
비교가 신뢰할 수 있게 된다.

사용법은 백엔드와 무관하게 한 줄이다.

    from common.llm import LLM
    llm = LLM()                     # 실행 환경을 스스로 판별한다 (common/runtime.py)
    llm = LLM(provider="openai")    # 강제로 지정하고 싶을 때만

원본(vLLM 전용) 대비 달라진 점
------------------------------
* provider 자동 판별. 환경변수 없이도 동작한다.
* n>1 을 지원하지 않는 백엔드에서는 호출을 n회 반복해 동일한 Generation 을 만든다.
  덕분에 03번의 best-of-N 실험이 API 모드에서도 그대로 성립한다.
* seed 미지원 백엔드에서는 seed 를 조용히 빼고, 결과에 deterministic=False 를 남긴다.
* 추론 모델(o 계열)의 max_tokens → max_completion_tokens 치환.
* 429(레이트리밋)에 지수 백오프 + Retry-After 존중.
* Usage 에 cost_usd 를 추가했다. 로컬 모델이면 None 이다(0 이 아니라 '모름/무료').
"""
from __future__ import annotations

import os
import random
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from . import pricing, runtime

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - 환경에 따라 미설치일 수 있다
    OpenAI = None  # type: ignore


# 원본과의 호환을 위해 남겨 둔 기본값. 실제 값은 runtime 이 정한다.
DEFAULT_BASE_URL = os.environ.get("VLLM_BASE_URL", runtime.DEFAULT_VLLM_BASE_URL)
DEFAULT_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
DEFAULT_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")


@dataclass
class Usage:
    """하나의 계측 범위에 대한 누적 집계."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    samples: int = 0
    seconds: float = 0.0
    cost_usd: float | None = None          # API 모드에서만 채워진다
    by_label: dict[str, dict] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, label: str, prompt_tokens: int, completion_tokens: int, samples: int,
            seconds: float, cost: float | None = None) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.calls += 1
        self.samples += samples
        self.seconds += seconds
        if cost is not None:
            self.cost_usd = (self.cost_usd or 0.0) + cost
        slot = self.by_label.setdefault(
            label, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0,
                    "samples": 0, "seconds": 0.0, "cost_usd": None}
        )
        slot["prompt_tokens"] += prompt_tokens
        slot["completion_tokens"] += completion_tokens
        slot["calls"] += 1
        slot["samples"] += samples
        slot["seconds"] += seconds
        if cost is not None:
            slot["cost_usd"] = (slot["cost_usd"] or 0.0) + cost

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
            "samples": self.samples,
            "seconds": round(self.seconds, 2),
            "cost_usd": None if self.cost_usd is None else round(self.cost_usd, 6),
            "by_label": self.by_label,
        }


@dataclass
class Generation:
    """한 번의 생성 호출 결과."""

    texts: list[str]
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    label: str
    cost_usd: float | None = None
    deterministic: bool = True   # seed 를 실제로 걸었는지. False 면 재현이 보장되지 않는다

    @property
    def text(self) -> str:
        return self.texts[0] if self.texts else ""


class LLM:
    """OpenAI 호환 엔드포인트(vLLM / OpenAI) 래퍼. 사용량과 비용을 함께 계측한다."""

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, provider: str | None = None):
        if OpenAI is None:
            raise RuntimeError("openai 패키지를 import 할 수 없다. pip install openai 를 먼저 실행한다.")

        # 인자를 하나도 주지 않으면 실행 환경을 스스로 판별한다.
        cfg = runtime.resolve_llm(force=provider) if provider else runtime.get_runtime().llm

        self.provider = cfg.provider
        self.model = model or cfg.model
        self.base_url = base_url or cfg.base_url
        self._supports_n = cfg.supports_n
        self._supports_seed = cfg.supports_seed
        self._is_reasoning = cfg.is_reasoning
        self.resolve_reason = cfg.reason

        key = api_key or cfg.api_key
        # OpenAI 클라이언트는 base_url=None 이면 공식 엔드포인트를 쓴다.
        kwargs: dict[str, Any] = {"api_key": key or "EMPTY"}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self.client = OpenAI(**kwargs)
        self.usage = Usage()

    # ── 계측 ────────────────────────────────────────────────────────
    def reset_usage(self) -> None:
        self.usage = Usage()

    @contextmanager
    def track(self, label: str = "scope"):
        """이 블록 안의 호출만 담는 새 Usage 를 내주고, 끝나면 바깥 총계에 합친다."""
        before = self.usage
        self.usage = Usage()
        try:
            yield self.usage
        finally:
            scoped = self.usage
            self.usage = before
            self.usage.prompt_tokens += scoped.prompt_tokens
            self.usage.completion_tokens += scoped.completion_tokens
            self.usage.calls += scoped.calls
            self.usage.samples += scoped.samples
            self.usage.seconds += scoped.seconds
            if scoped.cost_usd is not None:
                self.usage.cost_usd = (self.usage.cost_usd or 0.0) + scoped.cost_usd
            for k, v in scoped.by_label.items():
                slot = self.usage.by_label.setdefault(
                    k, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0,
                        "samples": 0, "seconds": 0.0, "cost_usd": None}
                )
                for kk, vv in v.items():
                    if kk == "cost_usd":
                        if vv is not None:
                            slot["cost_usd"] = (slot["cost_usd"] or 0.0) + vv
                    else:
                        slot[kk] += vv

    # ── 생성 ────────────────────────────────────────────────────────
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        n: int = 1,
        stop: list[str] | None = None,
        seed: int | None = 0,
        label: str = "chat",
        retries: int = 4,
    ) -> Generation:
        """chat 엔드포인트를 호출하고 사용량을 기록한다. n개의 텍스트를 담아 돌려준다.

        백엔드가 n>1 을 지원하지 않으면 내부적으로 n회 반복 호출해 같은 모양의
        결과를 만든다. 호출부(03번 best-of-N 등)는 차이를 알 필요가 없다.
        """
        if n > 1 and not self._supports_n:
            return self._chat_repeated(messages, temperature, max_tokens, n, stop, seed, label, retries)
        return self._chat_once(messages, temperature, max_tokens, n, stop, seed, label, retries)

    def _build_params(self, messages, temperature, max_tokens, n, stop, seed) -> dict:
        """백엔드별 파라미터 규약 차이를 여기서 흡수한다."""
        params: dict[str, Any] = {"model": self.model, "messages": messages, "n": n}

        if self._is_reasoning:
            # 추론 모델은 max_tokens 대신 max_completion_tokens 를 받고,
            # temperature 를 고정값 외에는 거부한다.
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
            params["temperature"] = temperature

        if stop:
            params["stop"] = stop
        if seed is not None and self._supports_seed:
            params["seed"] = seed
        return params

    def _chat_once(self, messages, temperature, max_tokens, n, stop, seed, label, retries) -> Generation:
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                params = self._build_params(messages, temperature, max_tokens, n, stop, seed)
                t0 = time.time()
                resp = self.client.chat.completions.create(**params)
                dt = time.time() - t0

                texts = [c.message.content or "" for c in resp.choices]
                pt = getattr(resp.usage, "prompt_tokens", 0) or 0
                ct = getattr(resp.usage, "completion_tokens", 0) or 0
                cost = None if pricing.is_local(self.provider) else pricing.cost_usd(self.model, pt, ct)

                self.usage.add(label, pt, ct, len(texts), dt, cost)
                return Generation(texts=texts, prompt_tokens=pt, completion_tokens=ct,
                                  seconds=dt, label=label, cost_usd=cost,
                                  deterministic=self._supports_seed and seed is not None)
            except Exception as e:  # pragma: no cover - 네트워크 의존
                last_err = e
                # 파라미터 미지원이면 기능 플래그를 내리고 즉시 재시도한다.
                if self._degrade_on_error(e):
                    # n>1 이 막힌 경우에는 같은 요청을 반복해도 소용없다.
                    # 반복 호출 경로로 갈아탄다.
                    if n > 1 and not self._supports_n:
                        return self._chat_repeated(messages, temperature, max_tokens, n,
                                                   stop, seed, label, retries)
                    continue
                time.sleep(self._backoff(e, attempt))
        raise RuntimeError(f"chat 호출이 {retries}회 재시도 후에도 실패했다: {last_err}")

    def _chat_repeated(self, messages, temperature, max_tokens, n, stop, seed, label, retries) -> Generation:
        """n>1 미지원 백엔드용 우회: 시드를 바꿔 가며 n회 호출해 합친다."""
        texts: list[str] = []
        pt = ct = 0
        cost_total: float | None = None
        t0 = time.time()
        for i in range(n):
            s = None if seed is None else seed + i   # 같은 답만 n번 나오지 않게 시드를 흔든다
            g = self._chat_once(messages, temperature, max_tokens, 1, stop, s, label, retries)
            texts.extend(g.texts)
            pt += g.prompt_tokens
            ct += g.completion_tokens
            if g.cost_usd is not None:
                cost_total = (cost_total or 0.0) + g.cost_usd
        return Generation(texts=texts, prompt_tokens=pt, completion_tokens=ct,
                          seconds=time.time() - t0, label=label, cost_usd=cost_total,
                          deterministic=False)

    def _degrade_on_error(self, e: Exception) -> bool:
        """'이 파라미터는 지원 안 한다'류 에러면 해당 기능을 끄고 True 를 준다."""
        msg = str(e).lower()
        changed = False
        if self._supports_seed and "seed" in msg and ("unsupported" in msg or "not supported" in msg or "unrecognized" in msg):
            self._supports_seed = False
            changed = True
        if self._supports_n and re.search(r"\bn\b.*(unsupported|not supported)|only.*n=1", msg):
            self._supports_n = False
            changed = True
        if not self._is_reasoning and "max_completion_tokens" in msg:
            self._is_reasoning = True   # 추론 모델 규약으로 전환
            changed = True
        return changed

    @staticmethod
    def _backoff(e: Exception, attempt: int) -> float:
        """429 는 Retry-After 를 존중하고, 그 외에는 지수 백오프 + 지터."""
        retry_after = getattr(getattr(e, "response", None), "headers", {}) or {}
        try:
            ra = float(retry_after.get("retry-after", 0))
            if ra > 0:
                return min(ra, 30.0)
        except Exception:
            pass
        return min(1.5 * (2 ** attempt) + random.random(), 30.0)

    def complete(self, prompt: str, system: str | None = None, **kwargs: Any) -> Generation:
        """프롬프트와 선택적 시스템 메시지로 messages 를 만들어 주는 편의 함수."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

    def health(self) -> dict:
        """엔드포인트가 살아 있는지 확인하고 요약을 돌려준다. 안 되면 예외."""
        gen = self.complete("Reply with the single word: ready", max_tokens=8,
                            temperature=0.0, label="health")
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url or "(공식 엔드포인트)",
            "reply": gen.text.strip(),
            "deterministic": gen.deterministic,
            "cost_usd": pricing.format_usd(gen.cost_usd),
            "resolved_by": self.resolve_reason,
        }
