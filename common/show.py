"""콘솔 출력 스타일: 모든 노트북이 자기 데이터 흐름을 지면 위에서 서술하게 만든다.

이 모듈의 목표는 재현이 아니라 **판독**이다. 저장된 출력만 들여다보는 독자가
아무것도 다시 실행하지 않고 무슨 일이 있었는지 따라갈 수 있어야 한다. 그래서
라벨 붙은 구획, 키/값 줄, 잘라 낸 코드와 텍스트, 번호 붙은 단계, 지표, A/B 비교를
모든 노트북에서 똑같은 모양으로 찍는다. 서식이 노트북마다 다르면 저장된 출력을
가로로 비교할 수 없고, 그러면 시리즈 전체의 수치가 서로 대화하지 못한다.

한국어판 주의
------------
**출력 문자열은 영문 원본 그대로 둔다.** 실행된 노트북에 저장돼 있는 출력은
원저자의 실측 결과이고, 여기서 라벨만 한국어로 바꾸면 코드와 저장 출력이 어긋나
"이 수치가 정말 이 코드에서 나왔는가" 를 확인할 길이 사라진다. 한국어로 다시 쓴
것은 주석과 독스트링뿐이다.
"""
from __future__ import annotations

import json
import shutil
import textwrap
from typing import Any


def _width() -> int:
    # 100자 상한을 두는 이유는 터미널이 아니라 노트북이다. 폭이 넓은 셸에서 실행하면
    # 구분선이 200자로 늘어나 노트북 출력 셀에서 강제 줄바꿈되고, 나중에 그 출력을
    # 다른 실행 결과와 나란히 비교할 수 없게 된다. 폭을 고정해야 출력이 기록물이 된다.
    try:
        return min(shutil.get_terminal_size((100, 20)).columns, 100)
    except Exception:
        return 100


def rule(title: str = "", char: str = "=") -> None:
    """제목을 가운데 둔 전체 폭 구분선을 찍는다. 제목이 없으면 선만 찍는다."""
    w = _width()
    if not title:
        print(char * w)
        return
    title = f" {title.strip()} "
    pad = max(0, w - len(title))
    left = pad // 2
    right = pad - left
    print(char * left + title + char * right)


def head(title: str) -> None:
    """가장 굵은 구획 머리글. "여기서부터 다른 실험이다" 를 알리는 단위다."""
    print()
    rule(title, "=")


def sub(title: str) -> None:
    """한 단계 약한 소구획 머리글. 같은 실험 안의 단계 구분에 쓴다."""
    print()
    rule(title, "-")


def step(n: Any, msg: str) -> None:
    """번호가 붙은 단계 표시.

    루프가 몇 회차까지 돌았는지를 출력만 보고 셀 수 있게 하는 장치다. 01번의
    완료까지 반복(run until done) 루프처럼 같은 작업이 여러 번 나오는 실험에서,
    n 을 회차로 넘기면 저장된 출력이 그대로 실행 이력이 된다.
    """
    print(f"\n[step {n}] {msg}")


def kv(label: str, value: Any, width: int = 26) -> None:
    """정렬된 키/값 한 줄. width 는 값의 시작 열을 고정해 세로로 읽히게 하는 폭이다."""
    print(f"  {str(label) + ':':<{width}} {value}")


def bullet(msg: str) -> None:
    """불릿 한 줄. 판정이 아니라 관찰을 나열할 때 쓴다."""
    print(f"  - {msg}")


def show_dict(d: dict, title: str = "", max_chars: int = 2000) -> None:
    """dict 를 들여쓴 JSON 으로 찍는다. 너무 길면 잘라 낸다.

    default=str 이 핵심이다. Path·datetime 처럼 JSON 이 모르는 객체가 섞여 있어도
    직렬화 실패로 셀 전체가 멈추지 않는다. 출력은 기록물이므로, 값 하나 때문에
    실험 결과를 통째로 잃는 쪽이 훨씬 나쁜 결과다.
    """
    if title:
        sub(title)
    text = json.dumps(d, indent=2, default=str)
    print(truncate(text, max_chars))


def truncate(text: str, max_chars: int = 800) -> str:
    """max_chars 까지 자르되, 몇 글자가 숨겨졌는지 표시를 남긴 문자열을 돌려준다.

    잘린 사실을 표시로 남기는 것이 중요하다. 표시가 없으면 독자가 짧은 출력을 보고
    "모델이 여기까지만 답했다" 고 오독한다.
    """
    # ⚠️ 함정: 여기서 나온 문자열을 다시 파싱하거나 채점기에 넣으면 안 된다. 이 함수는
    #    사람이 읽을 출력을 만드는 용도이고, 잘린 코드는 문법이 깨져 있으므로
    #    eval.check_asserts 등에 그대로 넘기면 모델 실력과 무관한 실패가 잡힌다.
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    hidden = len(text) - max_chars
    return text[:max_chars] + f"\n... [{hidden} more chars truncated] ..."


def code(snippet: str, title: str = "code", max_chars: int = 1500) -> None:
    """코드 블록을 머리글과 함께 찍는다. 각 줄 앞에 `  | ` 를 붙여 틀처럼 보이게 한다.

    들여쓰기가 있는 코드는 일반 출력과 섞이면 어디까지가 코드인지 알 수 없다.
    세로 막대가 그 경계를 만든다.
    """
    sub(title)
    body = truncate(snippet.rstrip(), max_chars)
    for line in body.splitlines():
        print("  | " + line)


def text(body: str, title: str = "", max_chars: int = 1500, wrap: bool = False) -> None:
    """자유 텍스트를 찍는다. 머리글과 줄바꿈 처리는 선택이다.

    wrap=True 는 모델이 뱉은 긴 산문에만 쓴다. 줄 단위로 textwrap.fill 을 걸기
    때문에, 코드처럼 줄바꿈 자체가 의미인 텍스트에 쓰면 형태가 망가진다.
    """
    if title:
        sub(title)
    body = truncate(body, max_chars)
    if wrap:
        for para in body.split("\n"):
            print(textwrap.fill(para, width=_width()))
    else:
        print(body)


def metric(name: str, value: Any, unit: str = "") -> None:
    """이름 붙은 지표 한 줄. float 는 소수 4자리로 통일한다.

    자릿수를 호출부가 아니라 여기서 고정하는 이유가 있다. 노트북마다 서식이 다르면
    저장된 출력끼리 비교할 때 0.77 과 0.7670 이 다른 값처럼 보인다.
    """
    if isinstance(value, float):
        value = f"{value:.4f}"
    suffix = f" {unit}" if unit else ""
    print(f"  {name:<32} {value}{suffix}")


def compare(name: str, baseline: float, treatment: float, unit: str = "", higher_is_better: bool = True) -> dict:
    """A/B 비교 한 덩어리를 찍고, 그 차이를 기록용 dict 로 돌려준다.

    베이스라인·처치·절대 변화량·상대 변화량, 그리고 higher_is_better 를 반영한
    화살표와 판정을 함께 찍는다. **찍기만 하고 끝나지 않고 dict 를 돌려주는 것이
    설계의 핵심이다.** 화면에 보이는 판정과 runlog.save_metrics 로 저장되는 판정이
    같은 계산에서 나와야, 나중에 저장된 지표와 노트북 출력이 어긋나지 않는다.
    """
    # ⚠️ 함정: 지연시간·비용·토큰처럼 "낮을수록 좋은" 지표에 higher_is_better 기본값을
    #    그대로 두면 개선이 regression 으로, 악화가 improvement 로 뒤집혀 찍힌다.
    #    delta 값 자체는 맞기 때문에 표를 대충 훑으면 알아채기 어렵다.
    delta = treatment - baseline
    # 베이스라인이 0 이면 상대 변화량은 정의되지 않는다. 0 이나 무한대로 채우지 않고
    # nan 을 남기는 이유는, 채워 넣은 값이 표에서 진짜 측정치처럼 읽히기 때문이다.
    rel = (delta / baseline * 100.0) if baseline not in (0, 0.0) else float("nan")
    improved = (delta > 0) == higher_is_better and delta != 0
    arrow = "up" if delta > 0 else ("down" if delta < 0 else "flat")
    verdict = "improvement" if improved else ("regression" if delta != 0 else "no change")
    u = f" {unit}" if unit else ""
    print(f"  {name}")
    print(f"    baseline : {baseline:.4f}{u}")
    print(f"    treatment: {treatment:.4f}{u}")
    print(f"    delta    : {delta:+.4f}{u}  ({rel:+.1f} percent)  [{arrow}, {verdict}]")
    return {
        "name": name,
        "baseline": baseline,
        "treatment": treatment,
        "delta": delta,
        "relative_percent": rel,
        "improved": bool(improved),
    }


def diagram(name: str, caption: str = "") -> None:
    """미리 그려 둔 손그림 다이어그램(images/<name>.png)을 노트북 안에 인라인으로 넣는다.

    경로 참조가 아니라 IPython Image 로 **바이트를 박아 넣는다.** 그래야 노트북 파일
    하나만 다른 곳으로 옮겨도 그림이 그대로 보인다. 마크다운의 `![](images/...)` 는
    저장소 밖에서 열면 깨진다.

    파일이 없거나 노트북 밖에서 호출돼도 예외를 올리지 않고 안내 문구만 찍는다.
    그림 한 장 때문에 실험 셀이 중단되는 것이 더 큰 손해이기 때문이다.
    """
    import os
    from pathlib import Path

    # 프로젝트 루트를 이 파일 위치에서 되짚는다(common/ 은 루트 바로 아래에 있다).
    # 노트북을 어느 작업 디렉터리에서 실행하든 그림 경로가 같아야 하기 때문이다.
    # LEN_PROJECT_ROOT 를 먼저 보는 것은 runtime/runlog 와 같은 규약을 쓰기 위해서다.
    root = os.environ.get("LEN_PROJECT_ROOT") or str(Path(__file__).resolve().parent.parent)
    path = Path(root) / "images" / f"{name}.png"
    try:
        from IPython.display import Image, Markdown, display

        if path.exists():
            if caption:
                display(Markdown(f"*{caption}*"))
            display(Image(filename=str(path)))
        else:
            print(f"[diagram '{name}' not found at {path}]")
    except Exception as e:  # 노트북 컨텍스트가 아닐 때(순수 파이썬 실행 등)
        print(f"[diagram '{name}': {e}]")


def table(rows: list[dict], columns: list[str] | None = None, title: str = "") -> None:
    """dict 목록을 열 맞춘 표로 찍는다.

    columns 를 생략하면 첫 행의 키 순서를 따른다. 행마다 키 집합이 다르면 첫 행에
    없는 키는 그냥 빠지므로, 08번처럼 여러 패턴을 한 표로 모을 때는 columns 를
    명시하는 편이 안전하다. 빠진 값은 예외 없이 빈 칸으로 채운다.
    """
    if title:
        sub(title)
    if not rows:
        print("  (no rows)")
        return
    columns = columns or list(rows[0].keys())
    widths = {c: max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    header = "  " + "  ".join(f"{c:<{widths[c]}}" for c in columns)
    print(header)
    print("  " + "  ".join("-" * widths[c] for c in columns))
    for r in rows:
        print("  " + "  ".join(f"{str(r.get(c, '')):<{widths[c]}}" for c in columns))
