"""데이터셋 로더: 노트북들이 공유하는 소수의 레코드 형태로 정규화한다.

벤치마크마다 컬럼 이름과 채점 규약이 제각각이다. 그 차이를 노트북 본문에서
매번 흡수하면 실험 코드가 데이터 정리 코드에 파묻힌다. 그래서 이 모듈이
경계 역할을 한다. **밖에서 들어오는 형태는 다양하지만, 나가는 형태는 Problem
하나뿐이다.**

설계 의도
---------
* 네트워크 없이 도는 SMOKE 세트를 내장한다. 00번은 실제 데이터셋을 건드리기
  전에 이 세트로 "모델 → 코드 → 채점기" 파이프라인 전체가 도는지 먼저
  확인한다. 계측기가 고장 난 상태로 큰 실험을 돌리는 사고를 막는 장치다.
* 실제 로더는 `datasets` 를 함수 안에서 지연 임포트한다. 이 패키지는 무겁고
  환경에 따라 없을 수도 있어서, 모듈을 임포트하는 것만으로 실패하면 안 된다.
* 스키마가 까다로운 벤치마크(LiveCodeBench, BIRD, CodeContests, CrossCodeEval,
  BigCodeBench)는 `inspect_dataset()` 으로 실제 스키마를 확인한 뒤에만 구현한다.
  아직 확인하지 못한 로더는 추측으로 채우지 않고 `_not_yet()` 으로 막아 둔다.
  틀린 데이터로 조용히 잘못된 수치를 내는 것보다 즉시 실패하는 편이 낫다.

공통 코드 과제 형태 (Problem)
-----------------------------
  id, prompt, entry_point, kind in {"asserts", "io"}, tests, meta
    - kind "asserts": tests 는 해답 뒤에 이어 붙일 assert 문 블록(문자열)이다
    - kind "io"     : tests 는 (stdin, 기대 stdout) 쌍의 리스트다

  이 두 갈래는 채점 방식이 다르다. asserts 는 해답과 테스트를 한 파일로 이어
  실행하고, io 는 표준입력을 넣어 표준출력을 비교한다(common/eval.py 의
  check_asserts / check_io).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Problem:
    id: str
    prompt: str
    entry_point: str = ""
    kind: str = "asserts"  # "asserts" 또는 "io" — 채점 경로가 갈리는 지점이다
    tests: Any = ""
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# 00번용 자급식 스모크셋 (네트워크 불필요)
#
# 문제 자체가 쉬운 것은 의도다. 여기서 pass@1 이 낮게 나오면 모델이 약한 게
# 아니라 하네스가 깨진 것이다. 난이도 측정이 아니라 배선 점검이 목적이다.
# 마지막 하나만 kind="io" 인 것도 의도다. 두 채점 경로를 모두 지나가게 한다.
# --------------------------------------------------------------------------------------
SMOKE_PROBLEMS: list[Problem] = [
    Problem(
        id="smoke/two_sum_indices",
        prompt=(
            "Write a Python function `first_pair_summing(nums, target)` that returns a tuple "
            "(i, j) with i < j of the first pair of indices whose values sum to target, "
            "scanning j from left to right and i before it. Return None if no pair exists."
        ),
        entry_point="first_pair_summing",
        kind="asserts",
        tests=(
            "assert first_pair_summing([2,7,11,15], 9) == (0,1)\n"
            "assert first_pair_summing([3,2,4], 6) == (1,2)\n"
            "assert first_pair_summing([1,2,3], 100) is None\n"
        ),
    ),
    Problem(
        id="smoke/is_palindrome",
        prompt="Write a Python function `is_palindrome(s)` that returns True if s reads the same forwards and backwards ignoring case and non-alphanumeric characters.",
        entry_point="is_palindrome",
        kind="asserts",
        tests=(
            "assert is_palindrome('A man, a plan, a canal: Panama') is True\n"
            "assert is_palindrome('race a car') is False\n"
            "assert is_palindrome('') is True\n"
        ),
    ),
    Problem(
        id="smoke/run_length_encode",
        prompt="Write a Python function `rle(s)` that run-length encodes a string, returning e.g. 'aaabb' -> 'a3b2'. Empty string returns ''.",
        entry_point="rle",
        kind="asserts",
        tests=(
            "assert rle('aaabb') == 'a3b2'\n"
            "assert rle('') == ''\n"
            "assert rle('abc') == 'a1b1c1'\n"
        ),
    ),
    Problem(
        id="smoke/fib",
        prompt="Write a Python function `fib(n)` returning the n-th Fibonacci number with fib(0)=0, fib(1)=1.",
        entry_point="fib",
        kind="asserts",
        tests="assert fib(0)==0\nassert fib(1)==1\nassert fib(10)==55\n",
    ),
    Problem(
        id="smoke/io_sum",
        prompt="Read two integers from stdin separated by whitespace and print their sum.",
        entry_point="",
        kind="io",
        tests=[("2 3", "5"), ("100 -1", "99")],
    ),
]


def load_smoke() -> list[Problem]:
    """내장 스모크 문제를 돌려준다 (네트워크 불필요).

    원본 리스트를 그대로 주지 않고 `list(...)` 로 복사해서 준다. 노트북이
    받은 리스트를 정렬하거나 잘라 써도 모듈 전역 상태가 오염되지 않게 한다.
    """
    return list(SMOKE_PROBLEMS)


# --------------------------------------------------------------------------------------
# 스키마 조사기: 까다로운 벤치마크의 로더를 설계할 때 먼저 돌린다
# --------------------------------------------------------------------------------------
def inspect_dataset(name: str, config: str | None = None, split: str = "test", n: int = 1) -> dict:
    """몇 행만 읽어 컬럼 이름과 샘플을 보고한다. 정규화 로더를 설계하기 위한 도구다.

    까다로운 로더를 구현하기 전에 이걸 먼저 돌린다. 그래야 상상한 스키마가
    아니라 실제 스키마에 맞춰 로더를 쓰게 된다.
    """
    from datasets import load_dataset

    ds = load_dataset(name, config) if config else load_dataset(name)
    if split not in ds:
        split = list(ds.keys())[0]   # 요청한 split 이 없으면 첫 split 으로 대체한다
    d = ds[split]
    sample = d[0] if len(d) else {}
    # 값이 수십 KB 인 컬럼(테스트 케이스 뭉치 등)이 흔하다. 화면을 덮지 않게 300자로 자른다.
    preview = {k: (str(v)[:300]) for k, v in sample.items()}
    return {"name": name, "config": config, "split": split, "num_rows": len(d), "columns": list(d.features.keys()), "sample": preview}


# --------------------------------------------------------------------------------------
# 실제 로더 (노트북별로 VM 에서 실제 스키마를 확인하고 구현했다)
# --------------------------------------------------------------------------------------
def _not_yet(name: str):
    """미구현 로더를 명시적으로 막는다.

    # ⚠️ 함정: 이 자리를 "대충 그럴듯한 구현"으로 메우고 싶은 유혹이 크다.
    # 스키마를 추측해 짠 로더는 대개 예외 없이 돌아가고, 대신 조용히 빈
    # tests 나 엉뚱한 entry_point 를 만들어 낸다. 그러면 pass@1 이 0 에
    # 가깝게 나오는데 모델 탓인지 로더 탓인지 구분할 방법이 사라진다.
    # 확인 전까지는 즉시 실패하는 편이 훨씬 싸다.
    """
    raise NotImplementedError(
        f"Loader for {name} is implemented on the VM after inspecting the real schema with "
        f"data.inspect_dataset(...). See the notebook that uses it."
    )


def load_mbpp_plus(limit: int | None = None, seed: int = 0) -> list[Problem]:
    """EvalPlus MBPP+ 를 asserts 형 문제로 정규화한다.

    01(완료까지 반복)·03(생성자-검증자)·13(PR 베이비시터)·17(캡스톤)의 주력
    데이터셋이다.

    모델에게는 자연어 설명과 **써야 할 함수 이름만** 준다. 테스트는 주지
    않는다. 채점과 루프의 실행 피드백은 base test_list 로 한다. 이것이 고전적인
    자기 디버깅 설정이다. 모델이 테스트를 못 본 채 먼저 쓰고, 얕은 버그로 일부가
    깨지고, 실패한 assert 를 보여 주면 그중 일부를 고친다.

    함수 이름을 프롬프트에 박는 이유는 채점 가능성 때문이다. 테스트가
    `assert similar_elements(...)` 처럼 특정 이름을 부르므로, 이름이 어긋나면
    코드가 맞아도 NameError 로 전부 실패한다. 이름은 참조 해답에서 뽑는다.

    seed 는 shuffle 에만 쓴다. limit 로 앞 N개만 잘라 써도 늘 같은 부분집합이
    나와야 실험이 재현 가능해진다.
    """
    import random
    import re

    from datasets import load_dataset

    ds = load_dataset("evalplus/mbppplus")
    split = "test" if "test" in ds else list(ds.keys())[0]
    out: list[Problem] = []
    for r in ds[split]:
        m = re.search(r"def\s+(\w+)\s*\(", r.get("code", ""))
        if not m:
            continue          # 참조 해답에서 함수 이름을 못 뽑으면 채점이 불가능하다
        entry = m.group(1)
        tests = "\n".join(r.get("test_list", []))
        if not tests:
            continue          # 테스트 없는 행은 검증 가능한 신호를 못 준다
        prompt = f"{r['prompt'].strip()}\n\nName your function exactly `{entry}`."
        out.append(Problem(id=f"mbppplus/{r['task_id']}", prompt=prompt, entry_point=entry, kind="asserts", tests=tests, meta={}))
    random.Random(seed).shuffle(out)   # 전역 random 을 건드리지 않는 전용 인스턴스를 쓴다
    return out[:limit] if limit else out


def load_humaneval_plus(limit: int | None = None) -> list[Problem]:
    """EvalPlus HumanEval+ 를 asserts 형 문제로 정규화한다. 실데이터 스모크 점검용이다."""
    from datasets import load_dataset

    ds = load_dataset("evalplus/humanevalplus")
    split = "test" if "test" in ds else list(ds.keys())[0]
    rows = ds[split]
    out: list[Problem] = []
    for r in rows:
        entry = r.get("entry_point", "")
        test = r.get("test", "")
        # EvalPlus 의 `test` 는 check(candidate) 함수를 "정의만" 해 둔다. 호출 줄이
        # 없으면 실행해도 아무것도 검사하지 않고 그냥 통과한다. 그래서 호출 줄을
        # 직접 이어 붙인다. 이게 없으면 pass@1 이 1.0 으로 나온다.
        test_block = test + (f"\ncheck({entry})\n" if "def check" in test else "")
        out.append(
            Problem(
                id=r.get("task_id", f"humanevalplus/{len(out)}"),
                prompt=r.get("prompt", ""),
                entry_point=entry,
                kind="asserts",
                tests=test_block,
                meta={"canonical_solution": r.get("canonical_solution", "")},
            )
        )
        if limit and len(out) >= limit:
            break   # 여기는 셔플이 없으므로 원본 순서 앞에서부터 자른다
    return out


def load_livecodebench(limit: int | None = 40, difficulties=("medium", "hard"), release: str = "test6.jsonl", seed: int = 0) -> list[Problem]:
    """LiveCodeBench 코드 생성 문제를 stdin/stdout(io) 형 Problem 으로 정규화한다.

    인자 하나하나가 오염(contamination)과 난이도를 겨냥한 선택이다.

    * `release="test6.jsonl"` — 설치된 datasets 라이브러리가 더는 데이터셋의
      로더 스크립트를 실행하지 않아서, 원시 jsonl 을 직접 읽는다. test6 은
      release_v6 로 대회 날짜가 2024년 이후다. 그 이전에 학습된 모델에 대해
      정답이 학습 데이터에 섞여 있을 가능성을 줄인다.
    * `difficulties=("medium","hard")` — easy 를 빼는 이유는 단발 시도가 자주
      성공해 버리면 루프가 고칠 여지, 즉 개선 가능 표면이 사라지기 때문이다.
    * 공개 테스트가 전부 stdin 형인 문제만 남긴다. functional 형이 섞이면
      check_io 로 채점할 수 없다.

    채점은 공개 테스트로만 한다. 비공개 테스트는 이 파일에 없다.
    """
    import json
    import random
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("livecodebench/code_generation_lite", release, repo_type="dataset")
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    probs: list[Problem] = []
    for r in rows:
        if r.get("difficulty") not in difficulties:
            continue
        try:
            pub = json.loads(r["public_test_cases"])
        except Exception:
            continue          # 테스트가 JSON 문자열로 이중 인코딩돼 있다. 깨진 행은 버린다
        if not pub or any(t.get("testtype") != "stdin" for t in pub):
            continue          # 하나라도 stdin 이 아니면 채점 경로가 어긋난다
        tests = [(t["input"], t["output"]) for t in pub]
        probs.append(
            Problem(
                id=f"lcb/{r.get('question_id')}",
                prompt=r["question_content"],
                entry_point="",   # io 형에는 진입 함수가 없다. 프로그램 전체가 답이다
                kind="io",
                tests=tests,
                meta={"difficulty": r.get("difficulty"), "title": r.get("question_title", ""), "platform": r.get("platform")},
            )
        )
    random.Random(seed).shuffle(probs)
    return probs[:limit] if limit else probs


def load_sql_create_context(limit: int | None = 60, seed: int = 0) -> list[dict]:
    """스키마가 함께 들어 있는 text-to-SQL 데이터 (b-mc2/sql-create-context).

    02(스킬·컨텍스트 엔지니어링)가 쓴다. 각 항목은 질문, CREATE TABLE 형태의
    스키마, 정답 SQL 로 이뤄진다. **여기서 스키마가 곧 주입 가능한 스킬이다.**
    같은 질문을 스키마 없이 물었을 때와 주입하고 물었을 때의 차이가 그 노트북의
    실험 설계 전부다.

    parquet 로 받아지고 외부 DB가 필요 없어서 자급식으로 돈다. Problem 이 아니라
    dict 를 돌려준다. 코드 과제가 아니므로 entry_point·kind 가 의미가 없다.
    """
    import random

    from datasets import load_dataset

    ds = load_dataset("b-mc2/sql-create-context")
    split = "train" if "train" in ds else list(ds.keys())[0]
    d = ds[split]
    idx = list(range(len(d)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        r = d[i]
        # 단일 테이블 스키마만 남긴다. sqltools.build_db 가 합성 행을 채워 실행
        # 동등성을 재는데, JOIN 이 걸리면 무작위로 채운 키가 서로 맞지 않아
        # 정답 SQL 조차 빈 결과를 낸다. 그러면 채점이 아니라 잡음이 된다.
        if r["context"].lower().count("create table") != 1:
            continue
        out.append({"id": f"sql/{i}", "question": r["question"], "schema": r["context"], "gold": r["answer"]})
        if limit and len(out) >= limit:
            break
    return out


def load_bird(*args, **kwargs):
    _not_yet("BIRD text-to-SQL")


def load_codecontests(*args, **kwargs):
    _not_yet("CodeContests (deepmind/code_contests)")


def load_crosscodeeval(*args, **kwargs):
    _not_yet("CrossCodeEval / RepoBench")


def load_bigcodebench_hard(*args, **kwargs):
    _not_yet("BigCodeBench-Hard (bigcode/bigcodebench-hard)")


def load_gitbugs(project: str = "hbase", base: str | None = None, max_corpus: int = 4000, seed: int = 0):
    """한 프로젝트의 GitBugs 중복 탐지 데이터.

    11(이슈 트리아지)의 중복 제거 실험이 쓴다. 실제 버그 트래커에서 사람이
    "중복"이라고 표시한 쌍이 정답이므로, 재현율을 검증 가능한 신호로 쓸 수 있다.

    Returns (corpus, pairs):
      corpus: dict issue_id -> "summary. description" (검색 대상이 되는 건초더미)
      pairs : (query_id, duplicate_id) 리스트. 양쪽 리포트가 모두 존재하는 쌍만 담는다
    corpus 에는 모든 쌍의 구성원을 먼저 넣고, 나머지 리포트를 max_corpus 까지
    무작위로 채운다. **쌍의 구성원이 빠지면 정답이 건초더미에 없어져 재현율
    상한이 1 보다 작아지므로, 그 부분만은 표본 추출에서 제외한다.**

    한국어판 변경: base 기본값이 원저자 서버의 절대경로
    (`/ephemeral/hf/gitbugs`)로 고정돼 있어서 다른 환경에서는 무조건
    FileNotFoundError 였다. 이제 None 이면 runtime.data_root() 를 따라가고,
    `LEN_DATA_ROOT` 환경변수로 덮어쓸 수 있다. 호출부가 base= 를 명시하면
    (11번이 그렇게 부른다) 그 값이 그대로 쓰인다.
    """
    import random

    import pandas as pd

    # 순환 임포트를 피하려고 함수 안에서 임포트한다. 이 모듈은 datasets·pandas 도
    # 같은 방식으로 지연 임포트하므로 관례가 어긋나지 않는다.
    from . import runtime

    if base is None:
        base = str(runtime.data_root() / "gitbugs")

    # engine="python" + on_bad_lines="skip": 버그 설명에 줄바꿈과 따옴표가 그대로
    # 들어 있어 C 파서가 중간에 죽는다. 몇 행을 잃더라도 끝까지 읽는 쪽을 택한다.
    rep = pd.read_csv(f"{base}/{project}/{project}_bugs.csv", engine="python", on_bad_lines="skip")
    reports: dict[str, str] = {}
    for _, r in rep.iterrows():
        iid = str(r.get("Issue id", "")).strip()
        summ = str(r.get("Summary", "") or "")
        desc = str(r.get("Description", "") or "")
        txt = (summ + ". " + desc).strip()
        # pandas 의 결측치는 문자열로 바꾸면 "nan" 이 된다. 그걸 정상 id 로 오인하면
        # 코퍼스에 유령 문서가 들어간다. 길이 5 이하는 제목만 있고 알맹이가 없는 행이다.
        if iid and iid.lower() != "nan" and len(txt) > 5:
            reports[iid] = txt[:2000]   # 임베딩 모델 입력 길이와 비용을 함께 억제한다
    comb = pd.read_csv(f"{base}/{project}/{project}_bugs-combined.csv", engine="python", on_bad_lines="skip")
    pairs = []
    for _, r in comb.iterrows():
        a = str(r.iloc[0]).strip()
        # 두 번째 열에 중복 후보가 쉼표로 여러 개 들어 있는 행이 있다. 첫 번째만 쓴다.
        b = str(r.iloc[1]).strip().split(",")[0].strip()
        if a in reports and b in reports and a != b:
            pairs.append((a, b))
    members = {x for p in pairs for x in p}
    others = [i for i in reports if i not in members]
    random.Random(seed).shuffle(others)
    # 쌍 구성원은 전부 넣고, 남는 자리만 다른 리포트로 채운다. members 가 이미
    # max_corpus 를 넘으면 max(0, ...) 덕분에 음수 슬라이스로 뒤집히지 않는다.
    keep = list(members) + others[: max(0, max_corpus - len(members))]
    corpus = {i: reports[i] for i in keep}
    return corpus, pairs


def load_swebench_verified_mini(*args, **kwargs):
    _not_yet("SWE-bench-Verified-Mini")
