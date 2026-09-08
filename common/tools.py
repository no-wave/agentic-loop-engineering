"""파이썬 실행 툴과 아주 작은 ReAct 루프. 커넥터·툴 노트북(06)이 쓴다.

커넥터라는 원시 기능의 요점은 모델이 가중치에서 답을 지어내는 대신 **실제 도구에
닿게 하는 것**이다. 여기서 그 도구는 파이썬 실행기이고, 이것은 임의의 MCP 커넥터를
대신하는 최소 사례다. 산술이나 데이터 집계처럼 모델이 암산으로 자주 틀리는 일을
툴이 대신하면, 정답 여부가 모델의 자신감이 아니라 실행 결과로 결정된다.

ReAct 루프는 그 위에 얹은 얇은 제어 구조다. 모델이 python 블록을 내면 실행해
표준출력을 다시 넣어 주고, 'FINAL:' 로 시작하는 줄이 나오면 거기서 멈춘다.

06번은 이 모듈에서 python_tool 과 first_number 만 쓴다. react_solve 는 같은
툴을 여러 스텝에 걸쳐 쓰고 싶을 때를 위한 참고 구현이다.

# ⚠️ 함정: python_tool 은 모델이 생성한 코드를 그대로 실행한다. 임의 코드
# 실행이다. eval.run_program 은 임시 디렉토리를 만들어 cwd 로 삼고 타임아웃을
# 걸 뿐, 별도의 프로세스 격리를 하지 않는다. 즉 실행되는 코드는 노트북을 돌리는
# 사용자 권한을 그대로 물려받는다. 파일을 지우고 네트워크에 나가고 환경변수의
# API 키를 읽어 내는 것을 막는 장치가 없다. 신뢰할 수 없는 출처의 문제·프롬프트를
# 이 툴에 물릴 때는 컨테이너나 VM 같은 진짜 경계 안에서 돌려야 한다.
# 09번의 가드레일 논의가 다루는 위험이 바로 이 지점의 실물이다.
"""
from __future__ import annotations

import re

from . import agents
from . import eval as ev

REACT_SYSTEM = (
    "You can use a Python tool. To run code, output a single fenced python block; the tool runs "
    "it and returns its stdout. Use print() to see values. When you are sure, output a line "
    "starting with 'FINAL:' followed by only the answer."
)


def python_tool(code: str, timeout: float = 8.0) -> str:
    """코드를 실행하고 모델에게 되돌려 줄 한 덩어리 문자열을 만든다.

    반환값이 항상 문자열인 것이 핵심이다. 성공·실패·타임아웃 어느 쪽이든 루프는
    같은 자리에 같은 형태의 관측을 받아야 다음 스텝을 이어 갈 수 있다.

    timeout 기본값 8.0 초는 eval.run_program 의 기본값 10.0 초보다 짧다. 이 툴은
    대화 한 스텝 안에서 불리므로, 무한 루프 하나가 전체 대화를 붙잡고 있는 시간을
    채점용 실행보다 더 짧게 끊는다.
    """
    r = ev.run_program(code, timeout=timeout)
    if r.timed_out:
        return "(timed out)"
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    # 실패했을 때 stderr 의 마지막 300자만 준다. 트레이스백은 앞쪽이 프레임 목록이고
    # 정작 원인인 예외 메시지는 맨 끝에 있다. 앞을 자르는 것이 맞는 방향이다.
    # 동시에 컨텍스트가 트레이스백으로 채워지는 것을 막는 예산 장치이기도 하다.
    return out if r.ok else f"(error) {err[-300:]}"


def react_solve(llm, question: str, max_steps: int = 4, label: str = "react") -> str:
    """파이썬 툴을 쓰는 짧은 ReAct 루프를 돌린다. FINAL 답변 텍스트를 돌려준다.

    max_steps=4 는 시도 상한이다. 툴 출력이 매 스텝 대화록에 누적되므로 스텝 수가
    곧 토큰 비용이고, 상한이 없으면 결론을 못 내는 질문 하나가 예산을 다 태운다.

    temperature=0.0 은 탐욕적 디코딩이다. 여기서 재는 것은 창의성이 아니라
    "툴을 붙였을 때 정답률이 오르는가" 이므로, 실행 간 변동을 없애야 한다.
    """
    transcript = f"Question: {question}\n"
    for _ in range(max_steps):
        gen = llm.complete(transcript, system=REACT_SYSTEM, temperature=0.0, max_tokens=400, label=label)
        text = gen.text or ""
        if "FINAL:" in text:
            # 첫 줄만 취한다. 모델이 답 뒤에 설명을 덧붙이는 일이 흔한데,
            # 그것까지 답으로 넘기면 정확 일치 채점이 통째로 실패한다.
            return text.split("FINAL:", 1)[1].strip().splitlines()[0].strip()
        code = agents.extract_code(text)
        # extract_code 는 코드 펜스가 없으면 원문 전체를 돌려준다. 그래서 산문을
        # 실행해 버리지 않도록 print 나 대입이 들어 있는지로 한 번 거른다.
        if code and ("print" in code or "=" in code):
            result = python_tool(code)
            transcript += text + f"\nTOOL OUTPUT:\n{result}\n"
        else:
            # 아무 말도 안 해 주면 모델이 같은 실수를 반복한다. 무엇이 잘못됐는지
            # 대화록에 남겨야 다음 스텝이 형식을 고칠 기회를 얻는다.
            transcript += text + "\n(no runnable code; output a python block or a FINAL line)\n"
    # 상한을 다 쓰고도 FINAL 이 없으면 마지막으로 결론만 직접 요구한다.
    # max_tokens=64 로 조인 것은 여기서 필요한 게 답 한 줄뿐이기 때문이다.
    g = llm.complete(transcript + "\nGive only FINAL: <answer>.", system=REACT_SYSTEM, temperature=0.0, max_tokens=64, label=label)
    t = g.text or ""
    # 그래도 FINAL 이 없으면 텍스트 전체를 답으로 넘긴다. 여기서 예외를 던지면
    # 형식을 못 맞춘 것과 답을 못 찾은 것이 뭉개진다. 채점기가 판단하게 둔다.
    return t.split("FINAL:", 1)[1].strip().splitlines()[0].strip() if "FINAL:" in t else t.strip()


def first_number(text: str):
    """텍스트에서 첫 번째 수를 뽑는다. 없으면 원문을 그대로 돌려준다.

    수치 답을 정확 일치로 채점할 때 "The answer is 42." 같은 응답이 틀린 것으로
    집계되는 일을 막는 최소한의 정규화다. 부호와 소수점을 함께 받는다.

    # ⚠️ 함정: 첫 번째 수를 무조건 집는다. "In 2024, the total was 42" 처럼
    # 답이 아닌 수가 앞에 오면 연도를 답으로 채점한다. 이런 느슨한 파서는
    # 툴을 붙인 쪽의 점수를 실제보다 높게도 낮게도 만들 수 있으므로, 실험
    # 결과를 읽을 때 파서 탓인 오차를 항상 함께 의심해야 한다.
    """
    m = re.search(r"-?\d+(?:\.\d+)?", text or "")
    return m.group() if m else (text or "").strip()
