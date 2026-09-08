"""채점기(scorer) pytest: eval.self_test 를 그대로 비추고 경계 사례를 몇 개 더 본다.

00번 노트북이 어떤 벤치마크보다 **먼저** `pytest common/tests` 를 돌린다. 순서가
논지다. 이 코스의 모든 수치는 "생성된 코드가 맞았는가" 라는 판정 위에 서 있으므로,
채점기가 틀리면 pass@1 도 개선폭도 전부 의미를 잃는다. 계측기를 먼저 계측한다.

self_test 가 이미 있는데 pytest 를 따로 두는 이유는 두 가지다. 첫째, self_test 는
assert 가 하나라도 깨지면 그 자리에서 멈춰 뒤쪽 채점기를 못 본다. pytest 는 함수를
독립적으로 돌려 **어느 채점기가 깨졌는지** 한 번에 보여 준다. 둘째, 여기서는
self_test 가 다루지 않는 경계값(단조성, 값의 범위, 런타임 에러 처리)을 따로 본다.
"""
import os
import sys

# 이 파일 기준으로 세 단계 위(= 프로젝트 루트)를 경로에 넣는다. 그래야 저장소
# 어느 위치에서 pytest 를 돌려도 common 패키지를 찾는다. 노트북 00번은 셸에서
# 바로 호출하므로 PYTHONPATH 가 잡혀 있다고 가정할 수 없다.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from common import eval as ev  # noqa: E402


def test_self_test_passes():
    """모듈 내장 자체 점검이 통과하는지부터 본다. 나머지 테스트의 전제다."""
    res = ev.self_test()
    assert res["all_passed"] is True


def test_check_io_correct_and_wrong():
    """맞는 프로그램과 틀린 프로그램을 **둘 다** 넣는다.

    맞는 쪽만 확인하면 "무조건 통과" 라고 답하는 고장 난 채점기를 잡지 못한다.
    채점기 검증에서는 오수용(false accept)을 잡는 반례가 정례보다 중요하다.
    """
    good = "import sys\na,b=map(int,sys.stdin.read().split())\nprint(a+b)"
    bad = "print(0)"
    # ⚠️ 함정: 이 테스트는 생성된 코드를 실제로 서브프로세스에서 실행한다. 채점기
    #    검증이라 입력이 고정돼 있어 안전하지만, 같은 채점기를 모델 출력에 쓰는 것은
    #    격리된 VM 안에서만 할 일이다. 로컬 개발 머신에서 벤치마크를 돌리면 신뢰할 수
    #    없는 코드를 자기 계정 권한으로 실행하는 셈이 된다.
    tests = [("2 3", "5"), ("4 4", "8")]
    assert ev.check_io(good, tests)["all_pass"] is True
    assert ev.check_io(bad, tests)["all_pass"] is False


def test_check_asserts_runtime_error_is_failure():
    """런타임 에러를 '실패' 로 셀 수 있어야 한다.

    assert 위반만 실패로 보고 ZeroDivisionError 같은 예외를 예외 처리로 흘려보내면,
    아예 실행되지 않는 코드가 통과로 기록된다. 판정 기준이 "예외 없이 끝났는가"
    하나여야 하는 이유가 이것이다.
    """
    code = "def f(x):\n    return x/0"
    assert ev.check_asserts(code, "assert f(1) == 1")["all_pass"] is False


def test_pass_at_k_monotonic_in_c():
    """정답 표본 수 c 가 늘면 pass@k 도 늘어야 한다.

    값의 정확도가 아니라 **부호와 방향**을 본다. pass@k 는 조합식이라 첨자 하나만
    틀려도 그럴듯한 숫자가 나오는데, 그런 실수는 대개 단조성을 깨뜨린다. 03번의
    best-of-N 비교는 이 단조성이 성립한다는 가정 위에서만 뜻이 있다.
    """
    assert ev.pass_at_k(10, 1, 1) < ev.pass_at_k(10, 5, 1) < ev.pass_at_k(10, 9, 1)


def test_edit_similarity_bounds():
    """같은 문자열은 정확히 1.0, 다른 문자열도 [0,1] 을 벗어나지 않아야 한다.

    정규화가 깨진 유사도는 지표를 비교 불가능하게 만든다. 상한이 1 이 아니면
    노트북별 유사도 수치를 나란히 놓을 수 없다.
    """
    assert ev.edit_similarity("abc", "abc") == 1.0
    assert 0.0 <= ev.edit_similarity("abc", "xyz") <= 1.0


def test_recall_and_map():
    """검색 지표는 순위가 섞인 입력으로 확인한다.

    첫 사례에서 정답은 {1, 2} 인데 상위 2개는 [3, 1] 이라 하나만 맞는다. 즉 0.5 다.
    이미 정렬된 목록을 넣으면 순위를 무시하는 구현도 통과하므로, 일부러 정답이
    아닌 3 을 맨 앞에 둔다. 04번 메모리·검색 증강의 Recall@k 가 이 계산에 기댄다.
    """
    assert ev.recall_at_k([3, 1, 2], {1, 2}, 2) == 0.5
    assert ev.average_precision([1, 2], {1, 2}) == 1.0
