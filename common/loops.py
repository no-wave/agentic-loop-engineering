"""루프 원시 요소: 여러 노트북이 공유하는 반복 엔진이다.

"스케줄링 + 완료까지 반복" 의 알맹이가 여기에 있다. 검증 가능한 정지 조건이 성립하거나
시도 상한에 닿을 때까지 같은 과제를 다시 시도하는 루프다.

**결정적인 설계 선택은 하나다. 시도와 시도 사이에 진짜 실행 피드백이 들어가는가.**
01번이 재는 것이 정확히 이 차이다. 외부 피드백 없는 자기 교정은 거의 도움이 되지 않고
(Huang 2023 의 결과), 테스트 실행 피드백이 붙은 자기 교정은 큰 격차를 메운다. 그래서
self_refine 은 use_feedback 플래그 하나로 두 조건을 갈라 놓았다. 나머지 조건이 전부
같아야 그 차이를 피드백의 기여로 읽을 수 있다.

12·13번과 17번 캡스톤도 이 self_refine 을 그대로 쓴다. 루프 몸통을 공유해야 노트북마다
달라지는 것이 과제와 검증 신호뿐이 된다.
"""
from __future__ import annotations

from . import agents
from . import eval as ev


def _io_feedback(problem, code: str, timeout: float = 10.0) -> str:
    """공개 테스트로 코드를 돌려 **첫 번째 실패 하나**를 모델에게 설명한다.

    실패를 전부 모아 주지 않는 것은 의도적이다. 앞선 실패가 뒤의 실패를 만드는 경우가
    대부분이라 목록을 늘려 봐야 컨텍스트만 커지고 모델의 주의는 흩어진다.

    실패 유형을 세 갈래로 나눠 서로 다른 문장을 만든다. 타임아웃은 stderr 가 비어 있어서
    그냥 넘기면 모델이 아무 단서도 못 받고, 예외와 출력 불일치는 고쳐야 할 부분이 아예
    다르기 때문이다. 이렇게 갈라 쓰는 것이 실행 피드백의 실체다.
    """
    for inp, exp in problem.tests:
        r = ev.run_program(code, stdin=inp, timeout=timeout)
        if r.timed_out:
            # 타임아웃에는 stderr 가 없다. 입력만 보여 주고 "느리다" 는 사실을 전한다.
            return f"On this input your program timed out:\n{inp}"
        if not r.ok:
            return f"On this input your program raised an error:\n{inp}\n\nstderr:\n{r.stderr[:500]}"
        got = ev._norm_output(r.stdout)
        if got != ev._norm_output(exp):
            return (
                f"On this input:\n{inp}\n\nexpected output:\n{exp}\n\n"
                # 출력이 길면 500자만 보인다. 빈 출력은 '(no output)' 으로 명시한다.
                # 빈 문자열을 그대로 붙이면 모델이 문장을 잘못 읽는다.
                f"but your program produced:\n{got[:500] if got else '(no output)'}"
            )
    return "All public tests passed."


def _assert_feedback(problem, code: str, timeout: float = 10.0) -> str:
    """후보를 assert 테스트에 걸어 실제 오류 출력을 그대로 회수한다.

    assert 형 문제(MBPP+, HumanEval+)의 실행 피드백이다. 캡처한 AssertionError 와
    트레이스백이 어느 케이스에서 무엇이 어긋났는지를 정확히 지목하고, 그 구체성이
    얕은 버그를 실제로 고치게 만든다. **모델이 스스로 원인을 추측하지 않아도 되게
    하는 것이 이 함수의 존재 이유다.**

    ⚠️ 함정: 여기서 넘어가는 문자열에는 정답 테스트의 내용이 실려 있다. 실험용으로는
    맞지만, 정답 테스트를 볼 수 없는 실전 루프에 이 피드백을 그대로 옮겨 놓고 같은
    개선폭을 기대하면 안 된다. 그 조건에서 무엇을 대신 쓸지가 03번의 주제다.
    """
    res = ev.run_program(code + "\n\n" + problem.tests, timeout=timeout)
    if res.ok:
        return "All tests passed."
    err = (res.stderr or "").strip()
    # 트레이스백은 **뒤에서** 700자를 자른다. 앞쪽은 호출 스택이고 정작 필요한 예외
    # 종류와 실패한 표현식은 맨 끝에 있기 때문이다. 앞에서 자르면 단서가 통째로 날아간다.
    return f"Running your function against the tests produced this error:\n{err[-700:] if err else '(timed out or no error text)'}"


def _refine_prompt(problem, code: str, feedback: str) -> str:
    """재시도 프롬프트: 원래 문제 + 직전 코드 + 피드백을 한 번에 준다.

    직전 시도를 대화 이력으로 넘기지 않고 매번 프롬프트에 다시 적어 넣는다. 시도가
    거듭돼도 컨텍스트가 선형으로 불어나지 않고, 각 시도가 같은 형태의 입력을 받아
    비교 가능해진다.
    """
    return (
        f"{agents.solve_prompt(problem)}\n\n"
        f"Your previous attempt was:\n```python\n{code}\n```\n\n"
        f"It is not correct. Test feedback:\n{feedback}\n\n"
        "Carefully fix the program. Return only one python code block with the corrected full program."
    )


def self_refine(llm, problem, max_attempts: int = 3, temperature: float = 0.0, use_feedback: bool = True, label: str = "refine") -> dict:
    """공개 테스트를 통과하거나 max_attempts 에 닿을 때까지 생성-테스트-수정을 반복한다.

    돌려주는 것: solved(성공 여부), attempts(실제 시도 횟수), solved_at(몇 번째에 풀렸는지,
    끝내 못 풀면 None), code(마지막 코드), history(시도별 정오 목록).

    solved 와 solved_at 을 따로 남기는 이유는 01번이 두 가지를 나눠 봐야 하기 때문이다.
    "결국 풀었는가" 와 "몇 번 만에 풀었는가" 는 다른 질문이고, 시도 상한을 얼마로 잡을지는
    후자의 분포가 결정한다. history 는 시도가 갈수록 나아지는지 아니면 맞았다 틀렸다를
    오가는지를 드러낸다.

    max_attempts 기본값 3 은 비용과 회수율의 절충이다. 상한을 늘려도 어느 지점부터는
    포화해 토큰만 더 든다.

    ⚠️ 함정: 정지 조건이 정답 테스트 통과다. 실전에서 이 자리에 "모델이 다 됐다고
    말했는가" 를 넣으면 루프는 언제나 성공으로 끝나고, 여기서 잰 개선폭은 재현되지
    않는다. 자기 개선 루프의 성능은 정지 조건의 검증 가능성을 절대 넘지 못한다.
    """
    history = []
    # 1회차는 피드백 없이 그냥 푼다. 이 결과가 곧 단일 시도 베이스라인이라
    # history[0] 만 모으면 별도 실행 없이 00번의 수치와 맞대볼 수 있다.
    code, gen = agents.single_shot(llm, problem, temperature=temperature, label=f"{label}-gen")
    res = agents.grade(problem, code)
    history.append(bool(res["all_pass"]))
    attempt = 1
    solved_at = 1 if res["all_pass"] else None

    while not res["all_pass"] and attempt < max_attempts:
        attempt += 1
        if use_feedback:
            fb = _io_feedback(problem, code) if getattr(problem, "kind", "asserts") == "io" else _assert_feedback(problem, code)
        else:
            # 피드백 없는 대조군. 틀렸다는 사실만 알리고 무엇이 틀렸는지는 주지 않는다.
            # 시도 횟수와 토큰 비용은 같게 유지되므로, 두 조건의 차이는 오직 정보량이다.
            fb = "Your previous solution was incorrect. Produce a different, correct solution."
        # max_tokens 를 1536 으로 올린다(1회차는 1024). 수정본은 원래 코드에 더해 방어
        # 로직이 붙으면서 길어지는데, 여기서 잘리면 문법 오류로 떨어져 피드백의 효과가
        # 아니라 토큰 상한을 측정하게 된다.
        gen = llm.complete(_refine_prompt(problem, code, fb), system=agents.SOLVE_SYSTEM, temperature=temperature, max_tokens=1536, label=f"{label}-fix")
        code = agents.extract_code(gen.text)
        res = agents.grade(problem, code)
        history.append(bool(res["all_pass"]))
        if res["all_pass"] and solved_at is None:
            solved_at = attempt

    return {
        "solved": bool(res["all_pass"]),
        "attempts": attempt,
        "solved_at": solved_at,
        "code": code,
        "history": history,
    }
