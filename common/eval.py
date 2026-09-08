"""채점과 격리 실행: 이 코스의 모든 수치가 딛고 서는 신뢰의 바닥이다.

생성된 코드가 맞았는지 틀렸는지를 판정하는 단일 출처가 이 모듈이다. 그래서 순서가
중요하다. **00번은 어떤 벤치마크도 돌리기 전에 self_test() 와 `pytest common/tests` 를
먼저 실행하고, 채점기가 그 자체 테스트를 통과한 뒤에야 아래로 나오는 pass@k·실행
정확도·유사도 수치를 신뢰한다.** 계측기를 검증하지 않고 얻은 숫자는 루프의 개선을 잰
것인지 채점 버그를 잰 것인지 끝내 구분할 수 없다.

검사 계열은 네 가지다.

  1. 표준입출력 프로그램 (CodeContests, LiveCodeBench): check_io
  2. 함수 + assert 형 문제 (HumanEval, MBPP, BigCodeBench): check_asserts
  3. text-to-SQL 실행 일치 (BIRD, Spider): sql_exec_match
  4. 코드 완성용 텍스트 유사도 (CrossCodeEval, RepoBench): exact_match, edit_similarity

여기에 분류 지표(prf1, recall_at_k, average_precision)가 더해진다. 09~11번과 15·16번의
트리아지·중복 제거 실험이 이 함수들로 채점된다.

생성된 코드는 임시 디렉터리 안에서, 타임아웃을 건 서브프로세스로 실행한다. 이것은
신뢰할 수 없는 모델 출력을 그대로 돌리는 일이며, 코스 전체가 격리된 VM 안에서
돌아간다는 전제에서만 허용된다.
"""
from __future__ import annotations

import difflib
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass


# --------------------------------------------------------------------------------------
# 서브프로세스 실행
# --------------------------------------------------------------------------------------
@dataclass
class RunResult:
    """한 번의 서브프로세스 실행 결과.

    ok 와 timed_out 을 따로 두는 이유는 두 실패가 성격이 다르기 때문이다. ok=False 는
    프로그램이 예외로 죽었다는 뜻이라 stderr 가 곧 실행 피드백이 되지만, timed_out 은
    무한 루프처럼 stderr 가 비어 있는 실패다. loops.py 가 이 둘을 갈라 모델에게 서로
    다른 문장을 돌려준다.
    """

    ok: bool
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    seconds: float


def run_program(code: str, stdin: str = "", timeout: float = 10.0) -> RunResult:
    """독립 실행형 파이썬 프로그램을 돌리고 stdin/stdout 을 주고받는다.

    임시 디렉터리를 만들어 그 안에 소스를 쓰고 cwd 까지 거기로 잡는다. 모델이 만든
    코드가 파일을 쓰더라도 저장소를 더럽히지 않고, 블록을 벗어나면 통째로 사라지게
    하기 위해서다. 타임아웃은 예외로 새어 나가지 않고 timed_out=True 인 RunResult 로
    바뀐다. 후보 하나가 무한 루프에 빠졌다고 채점 전체가 멈추면 안 되기 때문이다.
    """
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, path],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=d,          # 생성 코드가 쓰는 파일을 임시 디렉터리에 가둔다
            )
            dt = time.time() - t0
            return RunResult(
                ok=proc.returncode == 0,   # 종료 코드 0 만 성공. assert 실패는 1 로 잡힌다
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
                timed_out=False,
                seconds=dt,
            )
        except subprocess.TimeoutExpired:
            # ⚠️ 함정: timeout 은 "이 프로그램 1회 실행" 의 상한이지 채점 한 건의 예산이
            # 아니다. check_io 가 테스트 20개를 돌리면 최악의 경우 timeout×20 을 기다린다.
            # 후보 수 × 테스트 수 × timeout 으로 총 소요를 먼저 계산하고 값을 정한다.
            return RunResult(False, "", "timeout", -1, True, time.time() - t0)


def _norm_output(s: str) -> str:
    """비교 전에 출력을 정규화한다: 줄 끝 공백과 마지막 빈 줄을 없앤다.

    print 가 남기는 개행 하나 때문에 정답이 오답으로 뒤집히는 일을 막으려는 처리다.
    다만 줄 앞 공백은 건드리지 않는다. 들여쓰기가 의미를 갖는 출력이 있어서다.
    """
    lines = [ln.rstrip() for ln in s.replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def check_io(code: str, io_tests: list[tuple[str, str]], timeout: float = 10.0) -> dict:
    """표준입출력 프로그램을 (입력, 기대출력) 쌍들에 대해 채점한다.

    passed·total·all_pass 와 처음 실패한 케이스의 정보를 돌려준다. 부분 점수는 남기되
    **정답 판정은 전부 통과일 때만** 내린다. 하나라도 틀린 프로그램을 맞다고 받아들이면
    그 뒤의 모든 비교가 오수용(false accept) 위에 세워지기 때문이다.

    first_fail 은 처음 실패한 한 건만 담는다. 뒤이은 실패는 대개 같은 원인의 반복이라
    저장해 봐야 로그만 길어지고, 01번의 자기 개선 루프도 한 번에 한 가지 실패만
    모델에게 보여 준다.
    """
    passed = 0
    first_fail = None
    for i, (stdin, expected) in enumerate(io_tests):
        res = run_program(code, stdin=stdin, timeout=timeout)
        if res.ok and _norm_output(res.stdout) == _norm_output(expected):
            passed += 1
        elif first_fail is None:
            # stderr 를 400자로 자른다. 트레이스백 전문을 프롬프트에 넣으면 실행 피드백이
            # 컨텍스트를 잡아먹고, 정작 필요한 마지막 예외 줄은 대개 이 안에 들어온다.
            first_fail = {"index": i, "stderr": res.stderr[:400], "timed_out": res.timed_out}
    total = len(io_tests)
    return {
        "passed": passed,
        "total": total,
        # total > 0 조건이 핵심이다. 테스트가 하나도 없으면 "0개 전부 통과" 가 되어
        # 아무 코드나 정답이 된다. 빈 테스트 세트는 성공이 아니라 실패로 처리한다.
        "all_pass": passed == total and total > 0,
        "first_fail": first_fail,
    }


def check_asserts(code: str, test_block: str, timeout: float = 10.0) -> dict:
    """후보 코드 뒤에 assert 블록을 이어 붙여 실행한다. 종료 코드 0 이면 전부 통과다.

    HumanEval(끝에 'check(entry_point)' 를 붙이는 형태), MBPP(연속된 assert), 호출자가
    assert 하네스로 변환해 넘긴 BigCodeBench 계열 unittest 블록이 모두 이 한 가지
    방식으로 채점된다. 판정 기준이 "예외 없이 끝났는가" 하나뿐이라 데이터셋이 바뀌어도
    채점 의미가 흔들리지 않는다.

    ⚠️ 함정: 여기서 넘기는 test_block 은 신뢰할 수 있는 정답 테스트여야 한다. 03번처럼
    모델이 스스로 만든 테스트를 그대로 밀어 넣으면 통과 여부가 검증 신호가 아니라
    생성 품질의 반영이 되고, 그 수치로 변별력을 논할 수 없게 된다.
    """
    script = code + "\n\n" + test_block + "\n"
    res = run_program(script, timeout=timeout)
    return {"all_pass": res.ok, "stderr": res.stderr[:400], "timed_out": res.timed_out}


# --------------------------------------------------------------------------------------
# pass@k (Chen et al. 2021 의 불편 추정량)
# --------------------------------------------------------------------------------------
def pass_at_k(n: int, c: int, k: int) -> float:
    """n개를 뽑아 c개가 맞았을 때의 pass@k 불편 추정값.

    n개 중 k개를 뽑아 전부 틀릴 확률을 여집합으로 구한다. 단순히 "앞의 k개가 맞았나" 로
    세면 표본 순서에 따라 값이 출렁이지만, 이 추정량은 같은 n·c 에서 항상 같은 값을
    준다. 00번과 03번의 best-of-N 비교가 재현 가능한 이유가 여기에 있다.

    ⚠️ 함정: k > n 이면 `n - c < k` 가 성립해 무조건 1.0 이 나온다. 표본을 5개만 뽑아
    놓고 pass@10 을 물으면 전 문제 만점이라는 허수가 잡힌다. k 는 반드시 n 이하로 쓴다.
    """
    if k <= 0:
        return 0.0
    if n - c < k:
        # 틀린 표본이 k개보다 적으면 어떤 k개를 뽑아도 정답이 하나는 섞인다 → 확률 1
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


# --------------------------------------------------------------------------------------
# 코드 완성용 유사도
# --------------------------------------------------------------------------------------
def exact_match(pred: str, gold: str) -> bool:
    """앞뒤 공백만 무시한 완전 일치. 가장 엄격하고 해석의 여지가 없는 지표다."""
    return pred.strip() == gold.strip()


def edit_similarity(pred: str, gold: str) -> float:
    """difflib 비율로 [0,1] 정규화한 유사도다(1.0 이면 동일).

    실행할 수 없는 코드 조각(완성 과제)에는 통과/실패를 물을 수 없어 부득이 쓰는
    대리 지표다. 값이 높다고 코드가 도는 것은 아니라는 점을 늘 함께 기억한다.
    """
    return difflib.SequenceMatcher(None, pred.strip(), gold.strip()).ratio()


# --------------------------------------------------------------------------------------
# SQL 실행 일치 (BIRD / Spider)
# --------------------------------------------------------------------------------------
def _run_sql(db_path: str, sql: str, timeout: float = 30.0):
    """SQLite DB 에 조회 질의를 던지고 행을 돌려준다. 벽시계 상한을 함께 건다.

    sqlite3 는 실행 중인 질의를 밖에서 끊을 방법이 없어서, 진행 콜백을 걸어 스스로
    포기하게 만든다. 콜백이 0 이 아닌 값을 반환하면 질의가 중단된다.
    """
    con = sqlite3.connect(db_path)
    deadline = time.time() + timeout

    def _guard():
        return 1 if time.time() > deadline else 0

    # 100000: 콜백 호출 간격(VM 명령 수). 너무 촘촘하면 정상 질의까지 느려지고,
    # 너무 성기면 폭주하는 질의를 제때 못 끊는다.
    con.set_progress_handler(_guard, 100000)
    try:
        cur = con.execute(sql)
        rows = cur.fetchall()
        return rows
    finally:
        con.close()


def sql_exec_match(db_path: str, pred_sql: str, gold_sql: str, timeout: float = 30.0) -> dict:
    """실행 정확도: 예측 SQL 과 정답 SQL 이 같은 결과 집합을 내는가.

    문자열이 아니라 실행 결과로 비교한다. 같은 답을 내는 SQL 은 무수히 많아서 텍스트
    일치로는 맞는 질의를 계속 오답 처리하게 된다. 행 순서를 무시한 집합 비교를 쓰는
    것도 BIRD·Spider 가 명시적 ORDER BY 가 없는 질의를 채점하는 방식과 맞추기 위해서다.

    정답 SQL 이 먼저 실패하면 후보를 아예 돌리지 않고 나간다. 그 경우 문제는 후보가
    아니라 데이터셋이나 DB 경로 쪽이므로, 오류 메시지를 자르지 않고 그대로 남긴다.

    ⚠️ 함정: 순서를 무시하기 때문에 "상위 3개를 순서대로" 같은 문제에서는 정렬이 틀린
    답도 통과한다(오수용). ORDER BY 가 채점 대상인 과제라면 이 함수를 그대로 쓰면 안 된다.
    """
    try:
        gold_rows = _run_sql(db_path, gold_sql, timeout)
    except Exception as e:
        # 정답 쪽 실패는 데이터셋·스키마 문제다. 진단이 필요하므로 자르지 않는다.
        return {"match": False, "error": f"gold failed: {e}"}
    try:
        pred_rows = _run_sql(db_path, pred_sql, timeout)
    except Exception as e:
        # 예측 쪽 실패는 흔하고 메시지가 길다. 200자로 잘라 로그를 지킨다.
        return {"match": False, "error": f"pred failed: {str(e)[:200]}"}
    # repr 로 바꾼 뒤 정렬한다. 행 안에 int 와 str 이 섞여 있으면 튜플끼리는 정렬할 수
    # 없지만 문자열끼리는 언제나 정렬되기 때문이다.
    match = sorted(map(repr, pred_rows)) == sorted(map(repr, gold_rows))
    return {"match": match, "pred_rows": len(pred_rows), "gold_rows": len(gold_rows)}


# --------------------------------------------------------------------------------------
# 분류 채점 보조 (트리아지, SATD, 중복 제거)
# --------------------------------------------------------------------------------------
def prf1(tp: int, fp: int, fn: int) -> dict:
    """정밀도·재현율·F1 을 한 번에 낸다. 09·10·15·16번의 판정이 전부 이 함수를 지난다.

    분모가 0 인 경우를 예외 대신 0.0 으로 처리한다. 소수 클래스가 한 건도 안 잡히는
    구간이 실험 중에 정상적으로 나오는데, 거기서 죽으면 배치 전체가 멈춘다.

    tp·fp·fn 을 결과에 함께 실어 보내는 이유는, F1 만 남기면 같은 점수가 "적게 잡고
    정확했다" 인지 "많이 잡고 뭉갰다" 인지 구분되지 않기 때문이다.
    """
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def recall_at_k(ranked_ids: list, relevant_ids: set, k: int) -> float:
    """상위 k개 안에 정답이 몇 할 들어왔는지. 11번의 중복 이슈 검색을 이걸로 잰다.

    ⚠️ 함정: 분모가 정답 전체 개수라서, 정답이 k개보다 많으면 완벽한 검색기라도 1.0 에
    닿을 수 없다. 정답이 20건인데 Recall@10 을 재면 상한이 0.5 다. 낮은 값을 보고 검색기
    탓을 하기 전에 정답 개수 분포부터 확인한다.
    """
    if not relevant_ids:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & relevant_ids) / len(relevant_ids)


def average_precision(ranked_ids: list, relevant_ids: set) -> float:
    """평균 정밀도. 정답을 몇 개 찾았는지가 아니라 얼마나 위에 올렸는지를 잰다.

    정답을 만날 때마다 그 지점까지의 정밀도(hits/i)를 더하고 정답 개수로 나눈다. 같은
    개수를 찾아도 상위에 배치한 순위가 더 높은 점수를 받는다. 순위를 사람이 위에서부터
    읽는 상황에서는 Recall@k 보다 이쪽이 체감에 가깝다.

    순위 목록에 아예 등장하지 않은 정답은 0점으로 잡힌다. 분모가 relevant_ids 전체라
    따로 처리하지 않아도 자동으로 감점된다.
    """
    if not relevant_ids:
        return 0.0
    hits = 0
    score = 0.0
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in relevant_ids:
            hits += 1
            score += hits / i
    return score / len(relevant_ids)


# --------------------------------------------------------------------------------------
# 자체 테스트: 어떤 벤치마크가 이 채점기를 믿기 전에 먼저 통과해야 하는 관문
# --------------------------------------------------------------------------------------
def self_test() -> dict:
    """정답이 확실한 입력과 오답이 확실한 입력을 모든 채점기에 통과시켜 본다.

    핵심은 **양쪽을 다 본다**는 것이다. 맞는 코드를 맞다고 하는지만 확인하면 "무조건
    통과" 라고 답하는 고장 난 채점기도 합격한다. 그래서 모든 항목이 good/bad 쌍으로
    되어 있고, bad 쪽이 False 로 나오는지를 함께 단언한다. 이 함수가 곧 채점기의
    변별력 시험이다.

    00번이 첫 벤치마크 이전에 이것을 실행하고, 17번 캡스톤도 시작 전에 다시 돌린다.
    실패하면 예외로 즉시 멈춘다. 조용히 False 를 돌려주면 아무도 안 보기 때문이다.
    """
    results = {}

    # 1. 표준입출력: 맞는 덧셈 프로그램과, 부호만 틀린 프로그램을 나란히 넣는다.
    #    오답을 일부러 "거의 맞게" 만드는 것이 요점이다. 엉뚱한 코드는 어떤 채점기도
    #    걸러낸다.
    good = "import sys\na,b=map(int,sys.stdin.read().split())\nprint(a+b)"
    bad = "import sys\na,b=map(int,sys.stdin.read().split())\nprint(a-b)"
    io_tests = [("2 3", "5"), ("10 4", "14")]
    r_good = check_io(good, io_tests)
    r_bad = check_io(bad, io_tests)
    assert r_good["all_pass"] is True, r_good
    assert r_bad["all_pass"] is False, r_bad
    results["check_io"] = {"good_all_pass": r_good["all_pass"], "bad_all_pass": r_bad["all_pass"]}

    # 2. 함수 + assert
    fn = "def add(a, b):\n    return a + b"
    ok = check_asserts(fn, "assert add(2, 3) == 5\nassert add(-1, 1) == 0")
    no = check_asserts(fn, "assert add(2, 3) == 6")
    assert ok["all_pass"] is True and no["all_pass"] is False
    results["check_asserts"] = {"good": ok["all_pass"], "bad": no["all_pass"]}

    # 3. 무한 루프가 채점을 멈춰 세우지 않고 실패로 잡히는지 확인한다.
    #    timeout=2.0 은 기본 10초를 그대로 쓰면 자체 테스트가 매번 10초씩 늘어지기
    #    때문에 줄인 값이다. 걸리는지만 보면 되므로 2초로 충분하다.
    slow = "while True:\n    pass"
    r_slow = check_io(slow, [("", "")], timeout=2.0)
    assert r_slow["all_pass"] is False
    results["timeout_caught"] = True

    # 4. pass@k 의 양 끝과 중간값을 확인한다. n=5, c=1, k=1 이면 이론값이 정확히 1/5 이라
    #    0.19~0.21 구간을 요구하는 것으로 부동소수점 오차까지 감안한 검사가 된다.
    assert abs(pass_at_k(1, 1, 1) - 1.0) < 1e-9
    assert abs(pass_at_k(1, 0, 1) - 0.0) < 1e-9
    assert pass_at_k(5, 1, 1) > 0.19 and pass_at_k(5, 1, 1) < 0.21
    results["pass_at_k"] = {"n5_c1_k1": round(pass_at_k(5, 1, 1), 4)}

    # 5. 유사도. 공백만 다른 쌍은 완전 일치로, 한 글자 다른 쌍은 1.0 미만으로 나와야 한다.
    assert exact_match("x = 1", " x = 1 ") is True
    assert edit_similarity("abcd", "abcd") == 1.0
    assert 0.0 <= edit_similarity("abcd", "abxd") < 1.0
    results["similarity"] = {"em": True, "edit": round(edit_similarity("abcd", "abxd"), 3)}

    # 6. SQL 실행 일치. 임시 파일에 만든 초소형 DB 로 검사한다. ORDER BY 만 다른 질의는
    #    같다고, 행이 빠진 질의는 다르다고 나와야 한다 — 순서 무시와 내용 비교를 동시에
    #    확인하는 최소 쌍이다.
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "t.db")
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE t(id INTEGER, name TEXT)")
        con.executemany("INSERT INTO t VALUES (?,?)", [(1, "a"), (2, "b")])
        con.commit()
        con.close()
        same = sql_exec_match(db, "SELECT name FROM t ORDER BY id", "SELECT name FROM t")
        diff = sql_exec_match(db, "SELECT name FROM t WHERE id=1", "SELECT name FROM t")
        assert same["match"] is True and diff["match"] is False
        results["sql_exec_match"] = {"same": same["match"], "diff": diff["match"]}

    # 7. 검색 지표. 정답 2건 중 1건만 상위 3개에 들어왔으니 재현율은 0.5 다.
    #    평균 정밀도는 손으로 계산한 (1/1 + 2/3)/2 와 맞아야 한다.
    assert recall_at_k([1, 2, 3], {2, 9}, 3) == 0.5
    assert abs(average_precision([2, 1, 3], {2, 3}) - ((1 / 1) + (2 / 3)) / 2) < 1e-9
    results["ir_metrics"] = True

    results["all_passed"] = True
    return results


if __name__ == "__main__":
    import json

    print(json.dumps(self_test(), indent=2))
