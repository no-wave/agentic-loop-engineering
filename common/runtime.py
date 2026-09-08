"""실행 환경 자동 판별: 어떤 LLM 백엔드와 임베딩 모델을 쓸지 스스로 정한다.

이 모듈의 목표는 하나다. **학습자가 환경변수를 한 줄도 설정하지 않아도 노트북이
그대로 실행되는 것.** GPU 서버에서 vLLM을 띄워 뒀다면 그것을 쓰고, 노트북에서
OPENAI_API_KEY만 있다면 API를 쓴다. 둘 다 없으면 무엇을 하면 되는지 알려 준다.

판별 원칙
---------
1. **로컬 우선.** 로컬 vLLM이 살아 있으면 API 키가 있어도 로컬을 쓴다. 원본 코스의
   실측치가 자체 호스팅 모델 기준이고, 로컬은 토큰 비용이 0이라 사용자가 모르는
   사이에 과금될 위험이 없기 때문이다.
2. **강제 지정이 언제나 이긴다.** LLM_PROVIDER 를 명시하면 자동 판별을 건너뛰고,
   그 백엔드가 안 되면 조용히 다른 데로 넘어가지 않고 에러를 낸다. 조용한 폴백은
   "왜 이 수치가 나왔는지" 를 추적 불가능하게 만든다.
3. **LLM 과 임베딩은 따로 판별한다.** vLLM 은 로컬로 쓰면서 임베딩만 API 로 쓰는
   조합, 그 반대 조합이 모두 성립한다.
4. **판별 결과는 항상 보여 준다.** bootstrap() 배너로 무엇이 선택됐는지 출력한다.

환경변수 (전부 선택 사항)
------------------------
  LLM_PROVIDER      auto(기본) | vllm | openai
  EMBED_PROVIDER    미지정 시 LLM 판별 결과를 따라간다. local | openai
  VLLM_BASE_URL     기본 http://localhost:8000/v1
  VLLM_MODEL        비우면 서버가 서빙 중인 모델을 자동 조회한다
  VLLM_API_KEY      기본 EMPTY (vLLM 은 무시하지만 OpenAI 클라이언트가 값을 요구한다)
  OPENAI_API_KEY    API 모드에 필요
  OPENAI_MODEL      비우면 선호 순위표에서 자동 선택
  OPENAI_EMBED_MODEL 비우면 text-embedding-3-small
  OPENAI_BASE_URL   Azure/프록시/호환 게이트웨이를 쓸 때만
  LEN_PROJECT_ROOT  비우면 이 파일 위치에서 추정
  LEN_DATA_ROOT     외부 데이터 루트 (기본 <project>/data)
  LEN_REPROBE       1 이면 캐시를 무시하고 다시 탐지
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 상수: 라인업이 바뀌면 여기만 고친다
# ─────────────────────────────────────────────────────────────────────────────

# OpenAI 챗 모델 선호 순위. 앞에서부터 실제 사용 가능한 첫 모델을 채택한다.
# 기본값을 저비용 모델로 두는 이유는, 이 코스가 노트북 한 권당 수십~수백 회
# 생성을 돌리기 때문이다. 품질을 올리고 싶으면 OPENAI_MODEL 로 덮어쓰면 된다.
OPENAI_CHAT_PREFERENCE = [
    "gpt-5.6-luna",   # 기본: 저비용
    "gpt-5-nano",     # 더 저렴한 대안
    "gpt-5.6-terra",  # 성능/비용 균형
    "gpt-5.6-sol",    # 최상위
    "gpt-4o-mini",    # 구형 폴백
]

OPENAI_EMBED_PREFERENCE = [
    "text-embedding-3-small",  # 기본
    "text-embedding-3-large",
]

# 로컬 임베딩 모델: GPU 가 있으면 large, 없으면 small 로 자동 강등한다.
# bge-large 를 CPU 에서 돌리면 11번 노트북(코퍼스 4,000건)이 사실상 끝나지 않는다.
LOCAL_EMBED_GPU = "BAAI/bge-large-en-v1.5"
LOCAL_EMBED_CPU = "BAAI/bge-small-en-v1.5"

# BGE 계열은 비대칭 검색에서 쿼리 앞에 지시문을 붙여야 성능이 난다.
# OpenAI 임베딩에는 이런 규약이 없으므로 백엔드에 따라 켜고 끈다.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
PROBE_TIMEOUT_SEC = 1.5   # 로컬 서버 탐지 타임아웃. 길면 첫 셀이 답답해진다.
CACHE_FILENAME = ".len-runtime.json"


# ─────────────────────────────────────────────────────────────────────────────
# 판별 결과를 담는 자료구조
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    provider: str                 # "vllm" | "openai"
    model: str
    base_url: str | None
    api_key: str | None
    reason: str                   # 왜 이렇게 정해졌는지 (배너에 그대로 출력한다)
    probe_seconds: float = 0.0
    supports_n: bool = True       # n>1 다중 샘플 지원 여부
    supports_seed: bool = True    # seed 고정 지원 여부
    is_reasoning: bool = False    # o 계열 등 추론 모델이면 파라미터 규약이 다르다

    def redacted(self) -> dict:
        d = asdict(self)
        if d.get("api_key"):
            d["api_key"] = d["api_key"][:6] + "…"   # 키를 배너나 캐시에 그대로 남기지 않는다
        return d


@dataclass
class EmbedConfig:
    provider: str                 # "local" | "openai"
    model: str
    device: str | None            # local 일 때만 "cuda" | "cpu"
    reason: str
    query_instruction: str = ""   # BGE 는 지시문 필요, OpenAI 는 빈 문자열
    api_key: str | None = None
    base_url: str | None = None

    def redacted(self) -> dict:
        d = asdict(self)
        if d.get("api_key"):
            d["api_key"] = d["api_key"][:6] + "…"
        return d


@dataclass
class Runtime:
    llm: LLMConfig
    embed: EmbedConfig
    project_root: str
    data_root: str
    warnings: list[str] = field(default_factory=list)


class BackendNotFound(RuntimeError):
    """쓸 수 있는 백엔드를 하나도 찾지 못했을 때. 메시지 자체가 해결책이다."""


# ─────────────────────────────────────────────────────────────────────────────
# 경로
# ─────────────────────────────────────────────────────────────────────────────

def project_root() -> Path:
    """프로젝트 루트. 환경변수가 없으면 이 파일의 상위 디렉토리로 추정한다."""
    env = os.environ.get("LEN_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def data_root() -> Path:
    """외부 데이터 루트. 원본의 /ephemeral/hf 하드코딩을 대체한다."""
    env = os.environ.get("LEN_DATA_ROOT")
    return Path(env).expanduser().resolve() if env else project_root() / "data"


# ─────────────────────────────────────────────────────────────────────────────
# 탐지 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _http_get_json(url: str, timeout: float, api_key: str | None = None) -> dict | None:
    """GET 후 JSON 을 돌려준다. 실패하면 예외 대신 None 을 준다 (탐지용이므로)."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def probe_vllm(base_url: str | None = None, timeout: float = PROBE_TIMEOUT_SEC) -> tuple[bool, str | None, float]:
    """로컬 vLLM 이 살아 있는지 본다.

    Returns: (살아있음, 서빙 중인 첫 모델 id, 소요 초)
    """
    base = (base_url or os.environ.get("VLLM_BASE_URL") or DEFAULT_VLLM_BASE_URL).rstrip("/")
    t0 = time.time()
    payload = _http_get_json(f"{base}/models", timeout, os.environ.get("VLLM_API_KEY", "EMPTY"))
    dt = time.time() - t0
    if not payload:
        return False, None, dt
    data = payload.get("data") or []
    model_id = (data[0].get("id") if data and isinstance(data[0], dict) else None)
    return True, model_id, dt


def has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def list_openai_models(timeout: float = 5.0) -> list[str]:
    """실제로 쓸 수 있는 모델 목록. 조회에 실패하면 빈 리스트를 돌려준다."""
    base = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    payload = _http_get_json(f"{base}/models", timeout, os.environ.get("OPENAI_API_KEY"))
    if not payload:
        return []
    return [m.get("id", "") for m in (payload.get("data") or []) if isinstance(m, dict)]


def pick_preferred(available: list[str], preference: list[str], fallback: str) -> tuple[str, str]:
    """선호 순위표에서 사용 가능한 첫 모델을 고른다. Returns: (모델, 사유)"""
    if not available:
        # 목록 조회가 막힌 게이트웨이도 있다. 그럴 땐 1순위로 그냥 진행한다.
        return fallback, "모델 목록 조회 불가 → 기본값 사용"
    avail = set(available)
    for name in preference:
        if name in avail:
            return name, "선호 순위표에서 자동 선택"
    # 순위표에 아무것도 없으면 이름 접두사로 유사 모델을 찾아본다
    for name in preference:
        stem = name.split("-")[0] + "-" + (name.split("-")[1] if "-" in name[len(name.split("-")[0]):] else "")
        for a in sorted(avail):
            if a.startswith(stem):
                return a, f"'{name}' 미제공 → 유사 모델 '{a}' 대체"
    return fallback, "일치 모델 없음 → 기본값 사용"


def cuda_available() -> tuple[bool, str]:
    """torch 없이도 죽지 않게 감싼다. Returns: (사용가능, 장치명 또는 사유)"""
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return False, "torch 미설치"
    try:
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
        return False, "CUDA 미가용"
    except Exception as e:
        return False, f"CUDA 확인 실패: {e}"


def _is_reasoning_model(name: str) -> bool:
    """추론 모델은 max_tokens/temperature 규약이 다르다."""
    n = name.lower()
    return n.startswith(("o1", "o3", "o4")) or "-reasoning" in n


# ─────────────────────────────────────────────────────────────────────────────
# 핵심: 판별
# ─────────────────────────────────────────────────────────────────────────────

def resolve_llm(force: str | None = None) -> LLMConfig:
    """LLM 백엔드와 모델을 정한다. force 를 주면 자동 판별을 건너뛴다."""
    provider = (force or os.environ.get("LLM_PROVIDER") or "auto").strip().lower()

    # ── 강제 지정: vllm ──────────────────────────────────────────────
    if provider == "vllm":
        alive, served, dt = probe_vllm()
        if not alive:
            raise BackendNotFound(_msg_vllm_forced())
        model = os.environ.get("VLLM_MODEL") or served or "unknown"
        reason = "LLM_PROVIDER=vllm 강제 지정"
        return LLMConfig("vllm", model, _vllm_base(), os.environ.get("VLLM_API_KEY", "EMPTY"),
                         reason, dt)

    # ── 강제 지정: openai ────────────────────────────────────────────
    if provider == "openai":
        if not has_openai_key():
            raise BackendNotFound(_msg_openai_forced())
        return _resolve_openai("LLM_PROVIDER=openai 강제 지정")

    # ── 자동: 로컬을 먼저 본다 ──────────────────────────────────────
    if provider not in ("auto", ""):
        raise ValueError(f"LLM_PROVIDER 값이 잘못됐다: {provider!r} (auto|vllm|openai)")

    alive, served, dt = probe_vllm()
    if alive:
        model = os.environ.get("VLLM_MODEL") or served or "unknown"
        reason = f"{_vllm_base()} 응답 확인 ({dt:.1f}초)"
        cfg = LLMConfig("vllm", model, _vllm_base(), os.environ.get("VLLM_API_KEY", "EMPTY"),
                        reason, dt)
        # 사용자가 지정한 모델을 서버가 안 띄우고 있으면 실제 서빙 모델로 교정한다
        if os.environ.get("VLLM_MODEL") and served and os.environ["VLLM_MODEL"] != served:
            cfg.model = served
            cfg.reason += f" · VLLM_MODEL='{os.environ['VLLM_MODEL']}' 미서빙 → '{served}' 로 교정"
        return cfg

    if has_openai_key():
        return _resolve_openai(f"localhost 무응답 → OPENAI_API_KEY 감지")

    raise BackendNotFound(_msg_nothing_found())


def _vllm_base() -> str:
    return (os.environ.get("VLLM_BASE_URL") or DEFAULT_VLLM_BASE_URL).rstrip("/")


def _resolve_openai(reason: str) -> LLMConfig:
    explicit = os.environ.get("OPENAI_MODEL")
    if explicit:
        model, why = explicit, "OPENAI_MODEL 지정"
    else:
        model, why = pick_preferred(list_openai_models(), OPENAI_CHAT_PREFERENCE,
                                    OPENAI_CHAT_PREFERENCE[0])
    cfg = LLMConfig(
        provider="openai",
        model=model,
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        reason=f"{reason} · {why}",
        is_reasoning=_is_reasoning_model(model),
    )
    if cfg.is_reasoning:
        # 추론 모델은 temperature 고정, seed 무의미, n 미지원인 경우가 많다
        cfg.supports_seed = False
        cfg.supports_n = False
    return cfg


def resolve_embed(llm_cfg: LLMConfig | None = None, force: str | None = None) -> EmbedConfig:
    """임베딩 백엔드와 모델을 정한다. LLM 과 독립적으로 판별한다."""
    provider = (force or os.environ.get("EMBED_PROVIDER") or "").strip().lower()

    if provider == "openai":
        if not has_openai_key():
            raise BackendNotFound("EMBED_PROVIDER=openai 인데 OPENAI_API_KEY 가 없다.")
        return _embed_openai("EMBED_PROVIDER=openai 강제 지정")

    if provider == "local":
        return _embed_local("EMBED_PROVIDER=local 강제 지정")

    if provider not in ("", "auto"):
        raise ValueError(f"EMBED_PROVIDER 값이 잘못됐다: {provider!r} (auto|local|openai)")

    # ── 자동 판별 ────────────────────────────────────────────────────
    # LLM 이 로컬이면 임베딩도 로컬을 우선한다 (같은 GPU 를 이미 쓰고 있으므로).
    prefer_local = (llm_cfg is None) or (llm_cfg.provider == "vllm")

    if prefer_local:
        cfg = _embed_local("LLM 이 로컬이므로 임베딩도 로컬 우선")
        # sentence-transformers 가 아예 없으면 API 로 승격한다
        if not _has_sentence_transformers() and has_openai_key():
            return _embed_openai("sentence-transformers 미설치 → API 임베딩으로 전환")
        return cfg

    # LLM 이 API 인 경우: GPU 가 있으면 로컬 임베딩이 더 싸고 빠르다
    ok, dev = cuda_available()
    if ok and _has_sentence_transformers():
        return _embed_local(f"LLM 은 API 지만 CUDA({dev}) 가용 → 임베딩은 로컬")
    if has_openai_key():
        return _embed_openai("CUDA 미가용 → API 임베딩")
    return _embed_local("API 키 없음 → 로컬 임베딩(CPU)")


def _has_sentence_transformers() -> bool:
    try:
        import importlib.util  # noqa: PLC0415
        return importlib.util.find_spec("sentence_transformers") is not None
    except Exception:
        return False


def _embed_local(reason: str) -> EmbedConfig:
    ok, dev = cuda_available()
    device = "cuda" if ok else "cpu"
    model = os.environ.get("LOCAL_EMBED_MODEL") or (LOCAL_EMBED_GPU if ok else LOCAL_EMBED_CPU)
    detail = f"CUDA: {dev}" if ok else f"CPU ({dev})"
    if not ok and not os.environ.get("LOCAL_EMBED_MODEL"):
        detail += " → 경량 모델로 자동 강등"
    return EmbedConfig("local", model, device, f"{reason} · {detail}",
                       query_instruction=BGE_QUERY_INSTRUCTION)


def _embed_openai(reason: str) -> EmbedConfig:
    explicit = os.environ.get("OPENAI_EMBED_MODEL")
    if explicit:
        model, why = explicit, "OPENAI_EMBED_MODEL 지정"
    else:
        model, why = OPENAI_EMBED_PREFERENCE[0], "기본값"
    return EmbedConfig("openai", model, None, f"{reason} · {why}",
                       query_instruction="",  # OpenAI 임베딩은 쿼리 지시문 규약이 없다
                       api_key=os.environ.get("OPENAI_API_KEY"),
                       base_url=os.environ.get("OPENAI_BASE_URL"))


# ─────────────────────────────────────────────────────────────────────────────
# 캐시
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path() -> Path:
    return project_root() / CACHE_FILENAME


def _load_cache() -> dict | None:
    if os.environ.get("LEN_REPROBE") == "1":
        return None
    p = _cache_path()
    if not p.exists():
        return None
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    # 환경이 바뀌었으면 캐시를 버린다 (지문이 다르면 재탐지)
    if blob.get("fingerprint") != _fingerprint():
        return None
    return blob


def _save_cache(rt: Runtime) -> None:
    try:
        _cache_path().write_text(json.dumps({
            "fingerprint": _fingerprint(),
            "llm": rt.llm.redacted(),
            "embed": rt.embed.redacted(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass  # 캐시는 있으면 좋고 없어도 그만이다


def _fingerprint() -> str:
    """판별 결과에 영향을 주는 환경변수들의 지문. 하나라도 바뀌면 재탐지한다."""
    keys = ["LLM_PROVIDER", "EMBED_PROVIDER", "VLLM_BASE_URL", "VLLM_MODEL",
            "OPENAI_MODEL", "OPENAI_EMBED_MODEL", "OPENAI_BASE_URL", "LOCAL_EMBED_MODEL"]
    parts = [f"{k}={os.environ.get(k, '')}" for k in keys]
    parts.append(f"HAS_KEY={int(has_openai_key())}")
    return "|".join(parts)


_RUNTIME: Runtime | None = None


def get_runtime(refresh: bool = False) -> Runtime:
    """판별 결과를 얻는다. 프로세스 내에서는 1회만 탐지한다."""
    global _RUNTIME
    if _RUNTIME is not None and not refresh:
        return _RUNTIME

    cached = None if refresh else _load_cache()
    if cached:
        # 캐시에는 키가 마스킹돼 있으므로 실제 키는 환경변수에서 다시 채운다
        llm = LLMConfig(**{**cached["llm"], "api_key": _key_for(cached["llm"]["provider"])})
        llm.reason += " (캐시)"
        emb = EmbedConfig(**{**cached["embed"], "api_key": os.environ.get("OPENAI_API_KEY")})
        emb.reason += " (캐시)"
        _RUNTIME = Runtime(llm, emb, str(project_root()), str(data_root()))
        return _RUNTIME

    llm = resolve_llm()
    emb = resolve_embed(llm)
    rt = Runtime(llm, emb, str(project_root()), str(data_root()))

    if llm.provider == "openai":
        rt.warnings.append(
            "원본 코스의 수치는 Qwen2.5-Coder-32B(vLLM) 기준이다. API 모델에서는 값이 다를 수 있다."
        )
    if emb.provider == "local" and emb.device == "cpu":
        rt.warnings.append(
            "임베딩이 CPU 로 동작한다. 11번(코퍼스 4,000건)은 상당히 느리다. "
            "OPENAI_API_KEY 를 설정하면 API 임베딩으로 자동 전환된다."
        )
    _RUNTIME = rt
    _save_cache(rt)
    return rt


def _key_for(provider: str) -> str | None:
    return os.environ.get("VLLM_API_KEY", "EMPTY") if provider == "vllm" else os.environ.get("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────────────────────
# 부팅 배너
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap(estimate_usd: float | None = None, verbose: bool = True) -> Runtime:
    """노트북 첫 코드 셀에서 부른다. 판별하고, 무엇이 선택됐는지 보여 준다.

    자동화가 조용히 일어나면 학습자는 자기 수치가 어느 환경에서 나온 건지 알 수
    없게 된다. 그래서 배너 출력이 기본값이다.
    """
    try:
        rt = get_runtime()
    except BackendNotFound as e:
        print(str(e))
        raise

    if not verbose:
        return rt

    W = 66
    line = "━" * W
    print(line)
    print("  실행 환경 (자동 판별)")
    print(line)
    print(f"  LLM       : {rt.llm.provider:<8}  {rt.llm.model}")
    print(f"              ↳ {rt.llm.reason}")
    # 사유 문구에 이미 주소가 들어 있으면 중복 출력하지 않는다
    if rt.llm.provider == "vllm" and rt.llm.base_url and rt.llm.base_url not in rt.llm.reason:
        print(f"              ↳ {rt.llm.base_url}")
    caps = []
    if not rt.llm.supports_n:
        caps.append("n>1 미지원(반복 호출로 우회)")
    if not rt.llm.supports_seed:
        caps.append("seed 미지원(비결정적)")
    if caps:
        print(f"              ↳ {' · '.join(caps)}")

    dev = f"  ({rt.embed.device})" if rt.embed.device else ""
    print(f"  임베딩    : {rt.embed.provider:<8}  {rt.embed.model}{dev}")
    print(f"              ↳ {rt.embed.reason}")

    if rt.llm.provider == "vllm":
        print("  비용      : 없음 (자체 호스팅)")
    elif estimate_usd is not None:
        print(f"  비용 추정 : 이 노트북 전량 실행 시 약 ${estimate_usd:.2f}")

    print(f"  데이터    : {rt.data_root}")
    for w in rt.warnings:
        print(f"  ⚠️  {w}")
    print(f"  전환      : LLM_PROVIDER=vllm|openai  /  EMBED_PROVIDER=local|openai")
    print(line)
    return rt


# ─────────────────────────────────────────────────────────────────────────────
# 에러 메시지 (학습자가 가장 많이 막히는 지점이므로 해결책까지 같이 낸다)
# ─────────────────────────────────────────────────────────────────────────────

def _msg_nothing_found() -> str:
    return f"""
❌ 사용 가능한 LLM 백엔드를 찾지 못했다.

  [1] 로컬 vLLM  : {_vllm_base()} 에 응답이 없다
      → vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --port 8000
      → 다른 주소면  export VLLM_BASE_URL=http://<host>:<port>/v1
  [2] OpenAI API : OPENAI_API_KEY 가 설정돼 있지 않다
      → export OPENAI_API_KEY=sk-...   (또는 프로젝트 루트 .env 에 기재)

  둘 중 하나만 준비하면 나머지 설정은 자동으로 채워진다.
""".rstrip()


def _msg_vllm_forced() -> str:
    return f"""
❌ LLM_PROVIDER=vllm 로 강제 지정했으나 {_vllm_base()} 에 응답이 없다.

  → 서버 기동:  vllm serve <model> --port 8000
  → 주소 변경:  export VLLM_BASE_URL=http://<host>:<port>/v1
  → 자동 판별로 되돌리려면 LLM_PROVIDER 를 지우거나 auto 로 둔다.

  강제 지정한 백엔드가 없을 때 조용히 다른 백엔드로 넘어가지 않는 것은 의도된
  동작이다. 어떤 모델이 그 수치를 냈는지 추적할 수 없게 되기 때문이다.
""".rstrip()


def _msg_openai_forced() -> str:
    return """
❌ LLM_PROVIDER=openai 로 강제 지정했으나 OPENAI_API_KEY 가 없다.

  → export OPENAI_API_KEY=sk-...
  → 또는 프로젝트 루트의 .env 파일에 기재한다.
""".rstrip()


def describe() -> dict:
    """판별 결과를 dict 로. 노트북에서 show.show_dict() 로 찍기 좋다."""
    rt = get_runtime()
    return {
        "llm_provider": rt.llm.provider,
        "llm_model": rt.llm.model,
        "llm_reason": rt.llm.reason,
        "embed_provider": rt.embed.provider,
        "embed_model": rt.embed.model,
        "embed_device": rt.embed.device,
        "project_root": rt.project_root,
        "data_root": rt.data_root,
    }
