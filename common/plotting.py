"""A/B 막대와 곡선 그림을 일관된 모양으로 그려 results/figures 에 저장한다.

두 가지를 동시에 만족시켜야 한다.

1. 실행된 노트북이 차트를 **파일로** 남긴다. README 결과 표와 문서가 같은 그림을
   다시 쓰고, 그림이 어느 실행에서 나왔는지 파일로 추적된다.
2. 노트북 지면에도 그림이 **보인다.** 파일만 남으면 저장된 출력을 읽는 독자가
   차트를 못 본다. 그래서 save() 가 저장과 인라인 표시를 한 번에 한다.

matplotlib 은 모듈 최상단이 아니라 함수 안에서 늦게 import 하고 Agg 백엔드를 쓴다.
디스플레이가 없는 GPU 서버나 CI 에서도 이 모듈을 import 하는 것만으로는 절대
실패하지 않게 하려는 것이다. plotting 은 common 의 다른 모듈이 함께 끌려오는
자리에 있으므로, 여기서 나는 import 오류는 실험과 상관없이 노트북 첫 셀을 죽인다.

라벨·제목 문자열은 호출하는 노트북이 넘긴다. 이 모듈은 문자열을 만들지 않는다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .runlog import figures_dir


def _plt():
    """Agg 백엔드를 확정한 뒤 pyplot 을 돌려준다."""
    import matplotlib
    import koreanize_matplotlib

    # 순서가 의미다. pyplot 을 먼저 import 하면 백엔드가 이미 정해져 use("Agg") 가
    # 뒤늦게 먹지 않는 경우가 생긴다. 헤드리스 환경에서 창을 열려다 실패하지 않게
    # 항상 이 순서를 지킨다.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save(fig, name: str) -> Path:
    """그림을 results/figures 에 저장하고, **동시에** 노트북 지면에 인라인으로 띄운다.

    디스크의 파일과 지면의 그림 중 하나만 있으면 안 된다. 파일만 있으면 저장된
    노트북을 읽는 독자가 차트를 못 보고, 지면에만 있으면 문서·README 가 그 그림을
    다시 쓸 수 없다.
    """
    # ⚠️ 함정: results/figures 는 모든 노트북이 공유하는 한 폴더다. name 이 겹치면
    #    앞선 노트북의 그림을 경고 없이 덮어쓴다. 이름에 노트북 번호를 넣어
    #    "03_bestofn" 처럼 붙이는 관례를 지킨다.
    path = figures_dir() / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt

        # 닫지 않으면 노트북을 오래 돌릴수록 Figure 가 메모리에 쌓이고 matplotlib 이
        # 경고를 뱉는다. 저장이 끝난 뒤이므로 여기서 닫아도 잃는 것이 없다.
        plt.close(fig)
    except Exception:
        pass
    # 저장한 PNG 파일을 표시한다. Figure 객체가 아니라 파일을 넘기는 것이 핵심인데,
    # Agg 백엔드에서 맨 Figure 는 PNG 표현을 갖지 않아 인라인으로 렌더되지 않는다.
    try:
        from IPython.display import Image, display

        display(Image(filename=str(path)))
    except Exception:
        pass
    print(f"  saved + displayed figure -> {path}")
    return path


def ab_bar(name: str, labels: Sequence[str], values: Sequence[float], ylabel: str, title: str) -> Path:
    """지표 하나를 두 개 이상의 조건에서 비교하는 막대 그래프.

    베이스라인을 첫 번째 라벨로 넘기는 것이 관례다. 색 배열의 첫 색이 회색이라
    베이스라인이 시각적으로 뒤로 물러나고, 처치 조건이 앞으로 나온다.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    # ⚠️ 함정: 색은 4개뿐이다. 조건이 5개 이상이면 matplotlib 이 색을 되돌려 쓰기
    #    때문에 다섯 번째 막대가 첫 번째(베이스라인 회색)와 같은 색으로 그려진다.
    #    슬라이스가 에러를 내지 않고 조용히 통과하므로 그림을 눈으로 확인해야 한다.
    bars = ax.bar(list(labels), list(values), color=["#9aa0a6", "#3ee8c5", "#5b8def", "#f5a623"][: len(labels)])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # 막대 위에 값을 직접 찍는다. 눈금만 있으면 독자가 0.767 과 0.750 의 차이를
    # 읽지 못하고, 이 코스의 개선폭은 대개 그 정도 크기다.
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return save(fig, name)


def line(name: str, x: Sequence, series: dict[str, Sequence[float]], xlabel: str, ylabel: str, title: str) -> Path:
    """꺾은선 그래프. 회차별 pass@1 곡선이나 비용·품질 곡선에 쓴다.

    series 는 {계열 이름: y값들} 이고 x 는 모든 계열이 공유한다. 계열 길이가 x 와
    다르면 matplotlib 이 그 자리에서 에러를 내므로, 회차가 다른 실험을 한 그림에
    섞으려면 짧은 쪽을 미리 맞춰 두어야 한다.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # marker="o" 를 항상 켠다. 이 코스의 곡선은 표본점이 3~5개뿐이라, 점을 찍지
    # 않으면 실제로 측정한 지점과 선이 보간한 구간을 구별할 수 없다.
    for label, ys in series.items():
        ax.plot(list(x), list(ys), marker="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return save(fig, name)


def scatter_pareto(name: str, points: list[dict], xkey: str, ykey: str, labelkey: str, xlabel: str, ylabel: str, title: str) -> Path:
    """비용·품질 프런티어용 산점도. 점마다 그 전략 이름을 붙인다.

    08번(예산·관측)에서 "같은 품질을 더 싸게 내는 전략이 있는가" 를 눈으로 보는
    그림이다. xkey/ykey/labelkey 로 점 dict 의 어느 필드를 쓸지 넘기므로, 지표가
    바뀌어도 이 함수를 고칠 필요가 없다.
    """
    # ⚠️ 함정: 이름에 pareto 가 들어 있지만 이 함수는 프런티어를 계산하지 않는다.
    #    넘긴 점을 전부 그대로 찍을 뿐이므로, 지배당하는(더 비싸면서 더 나쁜) 전략도
    #    프런티어 위의 점처럼 보인다. 어느 점이 지배당하는지는 독자가 판단해야 한다.
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 5))
    for p in points:
        ax.scatter(p[xkey], p[ykey], s=60)
        ax.annotate(str(p[labelkey]), (p[xkey], p[ykey]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return save(fig, name)
