"""지표 저장과 구조화된 실행 로그.

모든 노트북은 끝에서 두 가지를 남긴다.

1. `results/<name>.json` — 그 노트북의 지표 전부 (save_metrics)
2. `results/loop-run-log.json` 에 덧붙이는 실행 항목 한 줄 (log_run)

2번이 쓰는 것은 운영 루프 공통 스키마다. run_id, pattern, duration_s, items_found,
actions_taken, escalations, tokens_estimate, outcome. 패턴이 이슈 트리아지든 CI
스위퍼든 항목의 모양이 같아야 서로 더할 수 있기 때문에 필드를 이렇게 고정했다.

**08번(예산·관측) 노트북이 이 로그를 다시 읽어 비용 모델을 실측 위에서 복원한다.**
08번 2부의 "패턴별 실측 토큰" 표와 그 뒤의 손익분기 계산은 전적으로 여기 쌓인
항목에서 나온다. 그래서 규칙이 하나 생긴다. **log_run 을 부르지 않은 노트북은
08번의 집계에서 통째로 사라진다.** 실행이 없었던 것과 기록하지 않은 것을 로그는
구분하지 못한다. README 의 결과 표 역시 results/*.json 에서 다시 생성되므로,
공개된 숫자는 전부 실제 실행까지 되짚을 수 있다.

한국어판이 추가로 기록하는 것
----------------------------
원본은 이 수치가 어느 백엔드에서 나왔는지 남기지 않았다. vLLM 전용이라 남길
필요가 없었다. 한국어판은 로컬 vLLM 과 OpenAI API 를 모두 지원하므로
(common/runtime.py 가 자동 판별한다) 같은 노트북을 돌려도 수치가 달라질 수 있다.
그래서 노트북들이 save_metrics 에 다음을 함께 실어 보낸다.

* `provider` / `model` — 어느 백엔드의 어느 모델이 낸 수치인지. 생성 호출이 없는
  노트북(10번 등)은 대신 임베딩 쪽 백엔드와 모델을 남긴다.
* `data_source` — 외부 데이터를 쓰는 노트북에서 실제 파일 경로인지 합성 폴백인지.
  이것이 없으면 데이터가 없어 폴백으로 돈 실행과 진짜 데이터 실행이 결과 파일에서
  똑같이 보인다.

save_metrics 는 넘어온 dict 를 그대로 직렬화하므로 스키마를 넓히려고 이 모듈을
고칠 필요가 없다. 08번에서 표가 서로 안 맞을 때 가장 먼저 봐야 할 것이 이 필드들이다.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    """노트북을 어느 디렉터리에서 실행하든 같은 프로젝트 루트를 가리키게 한다.

    상대 경로로 저장하면 결과 파일이 실행 위치마다 흩어지고, 08번이 읽을 로그가
    어디 있는지 알 수 없게 된다. 환경변수를 먼저 보는 것은 저장소를 옮기거나
    컨테이너 안에서 다른 경로로 마운트한 경우를 위한 탈출구다.
    """
    env = os.environ.get("LEN_PROJECT_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    # common/ 은 프로젝트 루트 바로 아래에 있다. 이 위치 가정이 깨지면 결과가
    # 엉뚱한 곳에 쌓이므로, 디렉터리 구조를 바꾼다면 여기도 같이 고쳐야 한다.
    return here.parent.parent


def results_dir() -> Path:
    """results/ 를 돌려준다. 없으면 figures/ 까지 함께 만든다.

    폴더 생성을 조회 함수 안에 넣어 둔 것은 의도적이다. 노트북이 실험 도중에 처음
    저장을 시도하는 순간 폴더가 없어 실패하면, 그때까지 돌린 모델 호출이 전부 날아간다.
    """
    d = _project_root() / "results"
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def figures_dir() -> Path:
    """그림 저장 폴더. plotting.save() 가 이 경로에만 쓴다."""
    d = results_dir() / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def now_iso() -> str:
    """UTC 초 단위 타임스탬프. 저장 시각이자 실행 항목의 run_id 로 쓰인다.

    로컬 시간을 쓰지 않는 이유는 명확하다. GPU 서버와 노트북 실행 환경의 타임존이
    다르면 로그의 시간 순서가 뒤섞이고, 08번이 실행을 시간축으로 정렬할 수 없다.
    """
    # ⚠️ 함정: 해상도가 초라서 같은 초 안에 log_run 을 두 번 부르면 run_id 가 겹친다.
    #    항목 자체는 둘 다 남지만 run_id 로 실행을 구분하는 코드는 둘을 하나로 본다.
    #    루프를 빠르게 여러 번 도는 실험이라면 pattern 이나 extra 로 따로 구분한다.
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_metrics(name: str, data: dict) -> Path:
    """지표 dict 를 results/<name>.json 에 쓰고 그 경로를 돌려준다.

    스키마를 강제하지 않는다. 노트북마다 재는 것이 다르고, 한국어판이 덧붙인
    provider/model/data_source 같은 필드도 이 자유로움 덕분에 모듈 수정 없이 들어간다.
    대신 무엇이 들어올지 모르므로 default=str 로 직렬화해서, Path 나 numpy 값이
    섞여도 저장 자체가 실패하지 않게 한다.

    _saved_at 은 setdefault 로 넣는다. 호출부가 직접 시각을 넣었다면 그쪽을 존중한다.
    """
    path = results_dir() / f"{name}.json"
    # ⚠️ 함정: 이 파일은 병합이 아니라 통째로 덮어쓴다. 노트북 앞부분만 다시 실행해
    #    일부 지표만 담아 저장하면, 앞선 완주 실행에서 남긴 나머지 필드가 사라진다.
    #    README 표는 이 파일에서 생성되므로 표의 칸이 조용히 비게 된다.
    payload = dict(data)
    payload.setdefault("_saved_at", now_iso())
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  saved metrics -> {path}")
    return path


def load_metrics(name: str) -> dict:
    """저장된 지표를 읽는다. 없으면 빈 dict.

    08번이 다른 노트북의 결과를 가져다 쓸 때 이 함수를 쓴다. 파일이 없을 때 예외
    대신 빈 dict 를 주는 것은, 학습자가 앞 노트북을 아직 안 돌렸어도 08번이
    거기서 멈추지 않고 "그 값은 비어 있다" 는 사실을 보여 주게 하기 위해서다.
    """
    path = results_dir() / f"{name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def log_run(
    pattern: str,
    duration_s: float,
    items_found: int = 0,
    actions_taken: int = 0,
    escalations: int = 0,
    tokens_estimate: int = 0,
    outcome: str = "report-only",
    extra: dict[str, Any] | None = None,
) -> dict:
    """운영 루프 스키마 항목 하나를 results/loop-run-log.json 에 덧붙인다.

    필드가 뜻하는 것은 이렇다.

    * `pattern` — 루프의 종류("maker-checker", "ci-sweeper" 등). 08번이 이 값으로
      묶어 합산하므로, 같은 패턴은 노트북이 달라도 철자를 맞춰야 한다.
    * `items_found` / `actions_taken` / `escalations` — 얼마나 찾았고, 자동 실행했고,
      사람에게 넘겼는지. 이 세 값이 09번 가드레일 논의의 분모가 된다.
    * `tokens_estimate` — 추정 토큰. 입력/출력 분할도 캐시 효과도 담지 못하는 거친
      계측기지만, 지출이 기록되었다는 사실 자체가 사후 감사를 가능하게 한다.
    * `outcome` — 기본값이 "report-only" 다. 아무것도 바꾸지 않은 실행이 기본이라는
      코스 전체의 입장이 기본값에 박혀 있다.

    호출부가 더 남길 것이 있으면 extra 로 넘긴다. 고정 스키마를 넓히지 않는 이유는,
    필드가 늘어나면 08번이 옛 항목과 새 항목을 같은 표에 올리지 못하기 때문이다.
    """
    # ⚠️ 함정: tokens_estimate=0 은 두 가지를 뜻할 수 있다. 정말 생성 호출이 없었던
    #    실행(10번의 로컬 분류처럼)과, 셀 계측을 빠뜨려 기록하지 않은 실행이다.
    #    08번은 둘을 합산해 버리므로, 0 을 보고 "값싼 패턴" 이라고 읽으면 틀린다.
    entry = {
        "run_id": now_iso(),
        "pattern": pattern,
        "duration_s": round(float(duration_s), 1),
        "items_found": int(items_found),
        "actions_taken": int(actions_taken),
        "escalations": int(escalations),
        "tokens_estimate": int(tokens_estimate),
        "outcome": outcome,
    }
    if extra:
        entry["extra"] = extra
    path = results_dir() / "loop-run-log.json"
    log = []
    if path.exists():
        try:
            log = json.loads(path.read_text())
        except Exception:
            # ⚠️ 함정: 파일이 깨져 있으면 빈 목록에서 다시 시작하고, 아래 write_text 가
            #    그 상태를 그대로 덮어쓴다. 즉 파싱 실패 한 번으로 과거 실행 기록이
            #    전부 사라지고 08번의 집계가 이 실행 하나로 줄어든다. 실험 중 커널을
            #    강제 종료해 파일이 잘렸다면, 다음 log_run 을 부르기 전에 백업해 둔다.
            log = []
    log.append(entry)
    path.write_text(json.dumps(log, indent=2, default=str))
    print(f"  logged run -> {path} ({pattern}: {outcome}, ~{tokens_estimate} tokens)")
    return entry


def load_run_log() -> list[dict]:
    """전체 실행 로그를 읽는다. 08번(예산·관측)의 실측 집계가 여기서 시작한다.

    로그가 없으면 빈 목록이다. 08번은 그 경우에도 1부의 비용 모형까지는 그대로
    돌아가고, 2부의 감사 표만 비게 된다.
    """
    path = results_dir() / "loop-run-log.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())
