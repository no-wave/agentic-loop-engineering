"""루프 엔지니어링 노트북 시리즈의 공통 유틸리티 (한국어판).

각 노트북은 필요한 모듈만 지연 임포트한다(예: `from common import eval, llm, show`).
패키지 __init__ 은 datasets, torch, vllm, sentence-transformers 같은 무거운 의존성을
일부러 임포트하지 않는다. 그래야 show.py 나 eval.py 같은 가벼운 모듈을 GPU 없는
환경에서도 그대로 쓰고 테스트할 수 있다.

한국어판에서 추가된 모듈
------------------------
  runtime : 실행 환경(LLM 백엔드·임베딩 모델)을 자동 판별한다
  pricing : 모델별 토큰 단가와 비용 계산
"""

__all__ = [
    "show",
    "eval",
    "llm",
    "runtime",
    "pricing",
    "runlog",
    "plotting",
    "data",
    "memory",
    "agents",
    "tools",
    "loops",
]
