"""재사용 가능한 에이전트 역할: 생성, 코드 추출, 채점.

노트북들이 루프로 조립해 쓰는 공용 부품 상자다. **여기 모아 두는 이유는 편의가 아니라
비교 가능성 때문이다.** 00번의 단일 에이전트 베이스라인, 01번의 자기 개선 루프, 03번의
생성자-검증자(maker-checker) 분리, 17번 캡스톤이 모두 같은 프롬프트로 생성하고 같은
채점기로 판정한다. 그래야 측정된 차이를 루프 구조의 기여로 읽을 수 있다. 노트북마다
프롬프트를 조금씩 손보면, 나중에 나온 +N점이 루프 덕인지 프롬프트를 다듬은 덕인지
아무도 증명할 수 없다.

이 파일의 프롬프트 문자열은 모델에 그대로 들어가는 입력이므로 영문 원문을 유지한다.
번역하면 원저자의 실측치와 다른 조건에서 실험하는 셈이 된다.

03번에서 쓰는 검증기(checker)는 세 갈래로 나뉘고, 그 변별력 차이가 그 노트북의 주제다.

  * 실행 기반  — select_by_tests, select_by_consistency, selftest_verify
  * 판단 기반  — judge_correct, judge_select (LLM 리뷰어, 실행하지 않는다)
  * 자기 신고  — self_assess (생성자 본인에게 물어본다. 믿으면 안 되는 쪽)
"""
from __future__ import annotations

import re

from . import eval as ev

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """펜스 코드 블록 중 마지막 것을 꺼낸다. 펜스가 없으면 원문을 그대로 돌려준다.

    첫 블록이 아니라 마지막 블록을 쓴다. 모델이 설명을 곁들이면서 중간에 예시 코드를
    보이고 끝에 완성본을 내는 패턴이 흔하기 때문이다. 펜스가 아예 없을 때 빈 문자열
    대신 원문을 넘기는 것도 의도된 선택이다. 채점기에서 문법 오류로 떨어지는 편이,
    조용히 빈 코드가 되어 "모델이 아무것도 못 했다" 로 집계되는 것보다 진단이 쉽다.
    """
    blocks = _FENCE.findall(text or "")
    if blocks:
        return blocks[-1].strip()
    return (text or "").strip()


# 모든 생성 호출이 공유하는 시스템 프롬프트. "펜스 블록 하나만" 을 못 박아 두어야
# extract_code 가 안정적으로 동작한다. 프롬프트 문자열은 모델 입력이므로 손대지 않는다.
SOLVE_SYSTEM = (
    "You are an expert Python programmer. Write correct, self-contained code. "
    "Return exactly one fenced python code block and nothing else."
)


def solve_prompt(problem) -> str:
    """문제 유형에 맞는 지시문을 만든다. 표준입출력형과 함수형은 요구가 다르다.

    kind 를 getattr 로 읽고 기본값을 "asserts" 로 둔 이유는, 데이터셋 어댑터가 이
    필드를 안 붙여도 다수파인 함수형으로 동작하게 하기 위해서다.
    """
    if getattr(problem, "kind", "asserts") == "io":
        return (
            "Solve this programming problem. Read input from standard input and write the "
            "answer to standard output.\n\n"
            f"{problem.prompt}\n\n"
            "Return only one python code block with the complete program."
        )
    return (
        "Implement the function described below. Return only one python code block with the "
        "complete function plus any imports or helpers it needs.\n\n"
        f"{problem.prompt}"
    )


def single_shot(llm, problem, temperature: float = 0.0, max_tokens: int = 1024, label: str = "solve"):
    """한 번의 풀이 시도를 생성한다. (코드, Generation) 을 돌려준다.

    temperature 기본값이 0.0 인 것은 베이스라인을 재현 가능하게 만들기 위해서다.
    같은 문제에 같은 답이 나와야 00번의 수치를 다시 뽑을 수 있다. 다양성이 필요한
    곳은 sample_n 처럼 호출부에서 명시적으로 온도를 올린다.

    label 은 llm 의 사용량 집계 키다. 셀마다 다른 이름을 주면 08번에서 어느 단계가
    토큰을 썼는지 분해해 볼 수 있다.
    """
    gen = llm.complete(
        solve_prompt(problem),
        system=SOLVE_SYSTEM,
        temperature=temperature,
        max_tokens=max_tokens,
        label=label,
    )
    return extract_code(gen.text), gen


def grade(problem, code: str, timeout: float = 10.0) -> dict:
    """숨겨진 정답 테스트로 풀이를 채점한다. eval 의 결과 dict 를 그대로 돌려준다.

    이 함수만이 정답 판정 권한을 갖는다. 아래의 검증기들은 정답 테스트를 절대 보지
    못하고, 그 검증기의 판정과 이 함수의 판정을 대조한 것이 03번의 변별력 수치다.
    """
    if getattr(problem, "kind", "asserts") == "io":
        return ev.check_io(code, problem.tests, timeout=timeout)
    return ev.check_asserts(code, problem.tests, timeout=timeout)


def is_correct(result: dict) -> bool:
    """결과 dict 에서 정답 여부만 뽑는다. 키가 없으면 False — 모르면 오답 쪽으로 센다."""
    return bool(result.get("all_pass"))


# --------------------------------------------------------------------------------------
# 생성자-검증자 구성 블록 (03번)
# --------------------------------------------------------------------------------------
def sample_n(llm, problem, n: int = 5, temperature: float = 0.8, max_tokens: int = 1024, label: str = "bon") -> list[str]:
    """후보 n개를 샘플링으로 뽑는다. best-of-N 에서 '생성자' 쪽을 맡는 함수다.

    temperature=0.8 은 다양성을 만들기 위한 값이다. 탐욕적 디코딩(temperature=0)으로
    n번 뽑으면 사실상 같은 답이 n개 나와서 검증기가 고를 것이 없어진다.

    ⚠️ 함정: n 을 키우면 비용이 정직하게 n배로 늘지만 정확도는 그만큼 오르지 않는다.
    검증기의 변별력이 상한이기 때문이다. 고르는 눈이 없으면 후보를 늘려도 오수용만
    같이 늘어난다. n 을 올리기 전에 검증기부터 점검한다.
    """
    out = []
    for _ in range(n):
        code, _ = single_shot(llm, problem, temperature=temperature, max_tokens=max_tokens, label=label)
        out.append(code)
    return out


_GEN_TESTS_SYSTEM = (
    "You write test cases for a Python function. Output only one python code block containing "
    "4 to 6 assert statements that call the function by its exact name. Do not solve the task."
)


def gen_self_tests(llm, problem, label: str = "gentests") -> str:
    """검증자가 명세만 보고 스스로 테스트를 쓴다(CodeT 계열). 숨겨진 정답 테스트는 못 본다.

    이 격리가 실험의 전부다. 검증기가 정답 테스트를 한 줄이라도 보면 그 뒤의 선택
    정확도는 검증 능력이 아니라 정답 유출을 측정한 값이 된다.

    시스템 프롬프트에서 "풀지 말고 assert 만" 을 명시하고 max_tokens 를 400 으로 묶는
    것도 같은 목적이다. 여유를 주면 모델이 테스트 대신 풀이를 쓰기 시작한다.
    """
    prompt = (
        "Write 4 to 6 assert statements that any correct solution to this task must satisfy. "
        "Use the exact function name. Do not implement the function, only the asserts.\n\n"
        f"{problem.prompt}"
    )
    g = llm.complete(prompt, system=_GEN_TESTS_SYSTEM, temperature=0.0, max_tokens=400, label=label)
    return extract_code(g.text)


def _count_passing(code: str, self_tests: str, timeout: float = 8.0) -> int:
    """후보가 검증자의 assert 를 몇 개 통과하는지 센다. 각 assert 를 개별로 감싼다.

    통과/실패의 이분법 대신 개수를 세는 이유는 후보들을 줄 세워야 하기 때문이다.
    assert 를 한 덩어리로 실행하면 첫 실패에서 멈춰 5개 중 4개 맞은 후보와 0개 맞은
    후보가 똑같이 '실패' 가 되고, 검증기는 고를 근거를 잃는다. 그래서 try/except 로
    하나씩 감싸 뒤 항목까지 전부 채점한다.
    """
    # 'assert' 로 시작하는 줄만 취한다. 모델이 함께 뱉는 주석·import·설명을 제외해야
    # 아래에서 만드는 try 블록의 들여쓰기가 깨지지 않는다.
    asserts = [ln for ln in self_tests.splitlines() if ln.strip().startswith("assert")]
    if not asserts:
        return 0
    harness = code + "\n_passed = 0\n"
    for a in asserts:
        harness += "try:\n    " + a.strip() + "\n    _passed += 1\nexcept Exception:\n    pass\n"
    harness += "print(_passed)\n"
    r = ev.run_program(harness, timeout=timeout)
    try:
        # 마지막 줄만 읽는다. 후보 코드가 자체적으로 print 를 하는 경우가 흔한데,
        # 우리가 심은 카운트는 언제나 맨 끝에 찍힌다.
        return int((r.stdout or "0").strip().splitlines()[-1])
    except Exception:
        # 후보가 임포트 단계에서 죽으면 stdout 이 비어 파싱이 실패한다. 이때 0점은
        # 정당하다. 실행조차 안 되는 코드는 어떤 테스트도 통과하지 못한 것이다.
        return 0


def select_by_tests(problem, candidates: list[str], self_tests: str, timeout: float = 8.0):
    """스스로 만든 테스트를 실행해 가장 많이 통과한 후보를 고르는 검증기다.

    (최고 인덱스, 점수목록) 을 돌려준다. 아무 후보도 단 한 개의 테스트를 통과하지 못하면
    0번으로 폴백한다. 그 상황에서는 검증기가 판단할 근거가 전혀 없으므로, 임의로 고르는
    대신 첫 후보를 쓰는 쪽이 결과를 재현 가능하게 만든다.

    ⚠️ 함정: scores 를 정확도로 읽으면 안 된다. 검증자가 쓴 테스트 자체가 틀릴 수 있어서,
    만점 후보가 오답인 경우가 실제로 나온다. 이 함수의 반환값은 '선택' 이지 '판정' 이
    아니며, 판정 권한은 숨겨진 테스트를 보는 grade() 에만 있다.
    """
    scores = [_count_passing(c, self_tests, timeout) for c in candidates]
    best = max(range(len(candidates)), key=lambda i: scores[i]) if any(scores) else 0
    return best, scores


def select_by_consistency(problem, candidates: list[str], timeout: float = 8.0):
    """표준입출력 문제용 실행 기반 검증기다(MBR-exec / 자기 일관성).

    모든 후보를 공개 테스트 입력으로 돌려 출력 서명(각 입력에 대한 출력의 튜플)을 만들고,
    같은 서명을 가진 후보가 가장 많은 집단에서 대표 하나를 뽑는다. **정답 출력은 한 번도
    보지 않는다.** 판단 근거는 오직 후보들끼리의 합의다. 기대출력 없이도 작동하기 때문에
    정답 테스트가 없는 실전 상황으로 옮기기 쉽다는 것이 이 방식의 장점이다.

    (최고 인덱스, 서명목록) 을 돌려준다.

    ⚠️ 함정: 합의는 정답의 증거가 아니다. 모델이 같은 오해를 공유하면 틀린 출력이 다수파가
    되고, 검증기는 그 오답을 자신 있게 고른다. 다수결이 강할수록 오수용도 함께 커진다는
    점이 03번에서 이 방식과 테스트 실행 방식을 굳이 나란히 재는 이유다.
    """
    from collections import Counter

    inputs = [inp for inp, _ in problem.tests]   # 기대출력은 의도적으로 버린다
    sigs = []
    for c in candidates:
        outs = []
        for inp in inputs:
            r = ev.run_program(c, stdin=inp, timeout=timeout)
            # 죽은 실행은 "<ERR>" 로 표시한다. 빈 문자열로 두면 아무것도 출력하지 않는
            # 정상 프로그램과 구분되지 않아 엉뚱한 합의 집단이 만들어진다.
            outs.append(ev._norm_output(r.stdout) if r.ok else "<ERR>")
        sigs.append(tuple(outs))
    # 전 입력에서 죽은 후보는 후보군에서 뺀다. 이걸 안 빼면 "다 같이 실패" 가 최대
    # 합의 집단이 되어 검증기가 전부 고장 난 코드를 고르게 된다.
    valid = [i for i, s in enumerate(sigs) if not all(o == "<ERR>" for o in s)]
    if not valid:
        return 0, sigs
    counts = Counter(sigs[i] for i in valid)
    best_sig = counts.most_common(1)[0][0]
    # 동률이면 앞선 인덱스가 이긴다. Counter 가 삽입 순서를 지키므로 결과가 재현된다.
    best = next(i for i in valid if sigs[i] == best_sig)
    return best, sigs


_JUDGE_SYSTEM = (
    "You are a strict code reviewer. Default to rejecting buggy code. You will see a task and "
    "several candidate solutions; choose the index of the one most likely to be fully correct."
)


def self_assess(llm, problem, code: str, label: str = "selfassess") -> bool:
    """구현자에게 자기 답이 맞았냐고 물어본다. **믿으면 안 되는 신호의 표본이다.**

    03번이 이 함수를 두는 목적은 쓰기 위해서가 아니라, 자기 신고가 검증을 대체할 수
    없다는 것을 수치로 보이기 위해서다. 13번의 PR 베이비시터도 같은 이유로 이 신호를
    대조군으로 쓴다.

    max_tokens=4 는 YES/NO 한 단어만 받기 위한 값이다. 이유를 설명할 여지를 주면 모델이
    스스로를 설득하며 긍정으로 기우는 경향이 있고, 토큰도 그만큼 더 든다.
    """
    g = llm.complete(
        f"Task:\n{problem.prompt}\n\nYour solution:\n```python\n{code}\n```\n\n"
        "Is this solution fully correct for all valid inputs? Answer only YES or NO.",
        temperature=0.0, max_tokens=4, label=label,
    )
    # 첫 글자가 Y 인지만 본다. "Yes"·"YES."·"Y" 를 모두 받으면서, 애매한 응답과
    # 빈 응답은 자동으로 부정 쪽에 떨어진다 — 판단 불가는 불합격으로 센다.
    return (g.text or "").strip().upper().startswith("Y")


# 리뷰어 시스템 프롬프트는 기본값을 NO 로 못 박는다. 그러지 않으면 LLM 리뷰어가
# 거의 모든 것을 승인해 버려서 변별력이 0 에 가까워진다.
_REVIEW_SYSTEM = "You are a strict code reviewer. Default to NO unless the code is clearly fully correct."


def judge_correct(llm, problem, code: str, label: str = "review") -> bool:
    """별도의 LLM 리뷰어가 코드를 읽고 정오를 판정한다. 실행은 하지 않는다.

    self_assess 와 짝을 이루는 대조군이다. 둘의 차이는 자기 코드를 보느냐 남의 코드를
    보느냐 하나뿐이고, 나머지 조건은 같게 맞춰 두었다. 실행 기반 검증기와 비교하면
    '읽어서 판단하기' 가 어디까지 가능한지가 드러난다.
    """
    g = llm.complete(
        f"Task:\n{problem.prompt}\n\nCandidate solution:\n```python\n{code}\n```\n\n"
        "Is it fully correct for all valid inputs? Answer only YES or NO.",
        system=_REVIEW_SYSTEM, temperature=0.0, max_tokens=4, label=label,
    )
    return (g.text or "").strip().upper().startswith("Y")


def selftest_verify(llm, problem, code: str, label: str = "selftest"):
    """스스로 테스트를 쓰고 **실제로 실행하는** 검증기. 전부 통과해야만 수용한다.

    judge_correct 와 달리 여기서는 판단의 근거가 텍스트가 아니라 실행 결과다. 기본
    태도가 거부라서, 애매하면 사람에게 넘어가고 오수용이 줄어든다.

    (수용여부, 생성된 테스트) 를 함께 돌려준다. 테스트 원문을 같이 넘기는 이유는,
    거부된 사례를 사후에 볼 때 후보가 나빴는지 테스트가 이상했는지 갈라야 하기 때문이다.

    ⚠️ 함정: 검증자가 assert 를 하나도 안 쓰면 True 를 반환한다. 거부할 근거가 없는
    상태를 '통과' 로 처리하는 셈이라, 이 경로가 잦아지면 수용률이 실력이 아니라
    테스트 생성 실패율을 반영하게 된다. 03번처럼 n_assert==0 비율을 따로 세어 둔다.
    """
    st = gen_self_tests(llm, problem, label=label)
    n_assert = sum(1 for ln in st.splitlines() if ln.strip().startswith("assert"))
    if n_assert == 0:
        return True, st  # 퇴화 사례: 테스트가 없으니 거부할 근거도 없다
    return _count_passing(code, st) == n_assert, st


def judge_select(llm, problem, candidates: list[str], label: str = "judge") -> int:
    """LLM 심판형 검증기(실행 없음): 후보 전부를 읽고 가장 맞을 법한 것의 인덱스를 고른다.

    select_by_tests 와 같은 자리에 꽂아 비교하기 위한 함수다. 입력이 같고 반환 형태만
    맞춰 두었으므로, 실행 기반과 판단 기반의 선택 정확도 차이를 그대로 잴 수 있다.
    """
    listing = "\n\n".join(f"[Candidate {i}]\n```python\n{c}\n```" for i, c in enumerate(candidates))
    prompt = (
        f"Task:\n{problem.prompt}\n\n{listing}\n\n"
        "Reply with only the integer index of the candidate most likely to be fully correct."
    )
    g = llm.complete(prompt, system=_JUDGE_SYSTEM, temperature=0.0, max_tokens=8, label=label)
    import re

    # 첫 번째 정수만 뽑는다. "Candidate 2" 처럼 답해도 받아 주기 위한 관용 처리다.
    m = re.search(r"\d+", g.text or "")
    idx = int(m.group()) if m else 0
    # 범위를 벗어난 인덱스는 0 으로 접는다. 모델이 없는 후보 번호를 부르는 일이 실제로
    # 있고, 여기서 막지 않으면 채점 루프 전체가 IndexError 로 무너진다.
    return idx if 0 <= idx < len(candidates) else 0
