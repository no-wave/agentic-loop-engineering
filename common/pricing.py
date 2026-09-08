"""모델별 토큰 단가와 비용 계산.

로컬 vLLM 에서 토큰은 '시간' 이지만 API 에서 토큰은 '돈' 이다. 08번(예산·관측)
노트북이 "품질의 토큰 가격" 을 다루므로, 그 서사를 API 모드에서도 그대로
성립시키려면 단가표가 필요하다.

단가는 자주 바뀐다. 본문 서술에 숫자를 박지 말고 항상 이 표를 참조한다.
출처: https://developers.openai.com/api/docs/pricing (2026-08 확인)
"""
from __future__ import annotations

# 1M 토큰당 USD, Standard 티어 기준.
# Batch/Flex 티어는 약 50% 할인이므로 BATCH_DISCOUNT 로 반영한다.
PRICES: dict[str, dict[str, float]] = {
    # ── OpenAI 챗 ────────────────────────────────────────────────
    "gpt-5.6-luna":  {"input": 0.20, "output": 1.20},   # 기본값
    "gpt-5-nano":    {"input": 0.05, "output": 0.40},
    "gpt-4o-mini":   {"input": 0.15, "output": 0.60},
    # sol / terra 단가는 확인 후 채운다. 없으면 UNKNOWN 처리되어 비용이 None 이 된다.

    # ── OpenAI 임베딩 (출력 토큰 없음) ───────────────────────────
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "text-embedding-ada-002": {"input": 0.10, "output": 0.0},
}

BATCH_DISCOUNT = 0.5   # Batch / Flex 티어

# 자체 호스팅 GPU 의 시간당 단가. 손익분기 분석(08번)에 쓴다.
# 클라우드 A100 80GB 온디맨드 기준의 대략치이며, 사용자가 자기 환경에 맞게 고친다.
GPU_HOURLY_USD = 2.00


def unit_price(model: str) -> dict[str, float] | None:
    """모델의 1M 토큰당 단가. 모르는 모델이면 None."""
    if model in PRICES:
        return PRICES[model]
    # 'gpt-5.6-luna-2026-08-01' 같은 날짜 접미사를 벗겨 다시 찾는다
    for name, p in PRICES.items():
        if model.startswith(name):
            return p
    return None


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int, batch: bool = False) -> float | None:
    """이 호출의 비용(USD). 로컬 모델이거나 단가를 모르면 None 을 준다.

    None 과 0.0 을 구분하는 것이 중요하다. 0.0 은 '공짜', None 은 '모름' 이다.
    """
    p = unit_price(model)
    if p is None:
        return None
    c = (prompt_tokens / 1_000_000) * p["input"] + (completion_tokens / 1_000_000) * p["output"]
    return c * (BATCH_DISCOUNT if batch else 1.0)


def is_local(provider: str) -> bool:
    return provider == "vllm"


def format_usd(v: float | None) -> str:
    if v is None:
        return "n/a"
    if v == 0:
        return "$0"
    if v < 0.01:
        return f"${v:.4f}"
    return f"${v:.2f}"


def breakeven_hours(api_cost_usd: float, gpu_hourly: float = GPU_HOURLY_USD) -> float | None:
    """이 API 비용이 자체 호스팅 GPU 몇 시간치인지. 08번 손익분기 분석용."""
    if gpu_hourly <= 0:
        return None
    return api_cost_usd / gpu_hourly
