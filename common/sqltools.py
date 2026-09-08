"""text-to-SQL 스킬 노트북(02)용 헬퍼.

데이터셋은 CREATE TABLE 스키마와 정답 SQL 을 함께 준다. 여기서 **스키마가 곧
주입 가능한 스킬**이다. 문제는 원본 데이터베이스가 없다는 것인데, 그래서 이
모듈은 DB 없이도 객관적으로 잴 수 있는 두 가지를 만든다.

  1. 스키마 유효성 — 예측 SQL 이 스키마에 실제로 존재하는 테이블·컬럼만
     참조하는가. 스키마를 주입하지 않으면 모델은 그럴듯한 이름을 지어낸다.
     그래서 이 값이 낮게 나온다. 의도 부채(intent debt)를 가장 깨끗하게
     드러내는 신호다.
  2. 실행 동등성 — CREATE TABLE 로 SQLite 테이블을 만들고 합성 행을 채운 뒤,
     정답 SQL 과 예측 SQL 을 모두 돌려 결과 집합을 비교한다. 무작위 데이터 위에서
     테스트 스위트를 돌리는 방식과 같은 발상이다.

두 지표는 성격이 다르다. 1번은 문자열만 보므로 값이 싸고 관대하다. 2번은 실제로
실행하므로 비싸고 엄격하다. 값싼 필터를 앞에 두고 비싼 검사를 뒤에 두는 캐스케이드
구조로 읽으면 된다.
"""
from __future__ import annotations

import os
import re
import sqlite3
import tempfile

from . import eval as ev

# 식별자처럼 생겼지만 스키마 이름이 아닌 토큰들. 이 목록이 빠지면 SELECT 나
# COUNT 같은 예약어가 "스키마에 없는 이름"으로 잡혀 유효성이 0 에 가까워진다.
_SQL_KEYWORDS = {
    "select", "from", "where", "group", "by", "order", "having", "count", "sum", "avg", "min",
    "max", "as", "and", "or", "not", "in", "like", "between", "join", "inner", "left", "right",
    "outer", "on", "distinct", "limit", "offset", "asc", "desc", "is", "null", "case", "when",
    "then", "else", "end", "union", "all", "exists", "into", "values", "create", "table", "int",
    "integer", "varchar", "text", "real", "float", "date", "datetime", "boolean", "primary",
    "key", "foreign", "references", "default",
    # 스키마 이름이 아닌 흔한 함수와 보조 토큰들
    "round", "abs", "length", "lower", "upper", "substr", "substring", "cast", "coalesce",
    "strftime", "now", "trim", "replace", "ifnull", "nullif", "total", "cnt", "avg_", "true",
    "false", "using", "cross", "natural", "rank", "row_number", "over", "partition", "with",
}


def schema_names(create_sql: str) -> set[str]:
    """CREATE TABLE 문에 선언된 테이블 이름과 컬럼 이름을 소문자 집합으로 돌려준다.

    SQL 파서를 쓰지 않고 정규식으로 처리한다. 이 데이터셋의 스키마가 한 줄짜리
    단순한 CREATE TABLE 로 정형화돼 있어서, 파서 의존성을 늘릴 이유가 없다.
    비교를 전부 소문자로 하는 것은 SQL 식별자가 대소문자를 가리지 않기 때문이다.
    """
    names: set[str] = set()
    for block in re.split(r"create\s+table", create_sql, flags=re.IGNORECASE):
        block = block.strip()
        if not block:
            continue          # split 결과의 맨 앞은 빈 문자열이다
        m = re.match(r"[`\"]?(\w+)[`\"]?\s*\((.*)\)", block, flags=re.DOTALL)
        if not m:
            # 괄호가 없거나 깨진 스키마. 최소한 테이블 이름만이라도 건진다.
            mt = re.match(r"[`\"]?(\w+)[`\"]?", block)
            if mt:
                names.add(mt.group(1).lower())
            continue
        names.add(m.group(1).lower())
        cols = m.group(2)
        for coldef in cols.split(","):
            cm = re.match(r"\s*[`\"]?(\w+)[`\"]?", coldef)
            if cm:
                names.add(cm.group(1).lower())   # 각 컬럼 정의의 첫 토큰이 컬럼 이름이다
    return names


def pred_identifiers(sql: str) -> list[str]:
    """스키마 이름과 대응해야 하는 식별자들. 키워드·리터럴·별칭은 뺀다.

    빼는 것들에는 각각 이유가 있다. 문자열 리터럴('Denver')은 데이터 값이지
    스키마 이름이 아니다. AS 뒤의 이름은 모델이 새로 만든 별칭이므로 스키마에
    없는 게 정상이다. 점 앞의 한정자는 별칭이거나 테이블 이름이다.
    이것들을 남겨 두면 정상적인 SQL 이 전부 무효로 찍힌다.

    # ⚠️ 함정: `\\b\\w+\\.` 규칙이 점 앞을 통째로 지우므로, 모델이
    # `bogus_table.name` 처럼 **없는 테이블 이름으로 한정한 참조는 검사되지
    # 않는다.** 즉 이 지표는 스키마 유효성을 과대평가하는 방향으로 치우친다.
    # 02번에서 "스키마 주입 후 유효성이 올랐다" 는 결론은 이 편향이 두 조건에
    # 똑같이 걸린다는 전제 위에서만 성립한다.
    """
    s = sql or ""
    s = re.sub(r"'[^']*'", " ", s)        # 작은따옴표 문자열 리터럴
    s = re.sub(r'"[^"]*"', " ", s)        # 큰따옴표 문자열 리터럴
    s = re.sub(r"(?i)\bas\s+\w+", " ", s)  # AS 뒤의 컬럼·테이블 별칭
    s = re.sub(r"\b\w+\.", " ", s)        # 점 앞의 별칭·테이블 한정자
    toks = re.findall(r"[A-Za-z_]\w*", s)   # 숫자로 시작하는 토큰은 식별자가 아니다
    return [t.lower() for t in toks if t.lower() not in _SQL_KEYWORDS]


def schema_validity(pred_sql: str, create_sql: str) -> bool:
    """예측 SQL 의 비키워드 식별자가 전부 스키마에 존재하면 True.

    하나라도 없으면 False 인 전부-아니면-전무 판정이다. 지어낸 컬럼이 하나만
    섞여도 그 질의는 실행되지 않으므로, 부분 점수를 주는 것이 오히려 현실을
    왜곡한다.
    """
    valid = schema_names(create_sql)
    ids = pred_identifiers(pred_sql)
    if not ids:
        return False   # 빈 응답이나 SQL 이 아닌 산문. 위반이 0건이라고 통과시키면 안 된다
    return all(i in valid for i in ids)


def _rand_value(coltype: str, i: int):
    """컬럼 타입에 맞는 합성 값을 만든다.

    이름은 rand 지만 실제로는 i 의 결정적 함수다. 같은 스키마를 몇 번 돌려도
    같은 DB 가 나와야 실행 동등성 판정이 실행마다 흔들리지 않는다.
    """
    t = (coltype or "").lower()
    if "int" in t or "real" in t or "float" in t or "numeric" in t or "decimal" in t:
        return (i * 7 + 3) % 100   # 7 은 100 과 서로소라 값이 겹치지 않고 퍼진다
    return f"v{i % 5}"             # 문자열은 5종만 만들어 GROUP BY 가 실제로 묶이게 한다


def build_db(create_sql: str, n_rows: int = 8) -> str:
    """CREATE TABLE 스키마로 임시 SQLite DB 를 만들고 합성 행을 채운다.

    실패를 삼키는 곳이 여러 군데다(테이블 생성, 행 삽입). 스키마 문자열이
    조금 깨져 있어도 나머지 테이블은 만들어 두고, 최종 판정은 exec_equiv 에서
    정답 SQL 이 도는지 여부로 내리게 하려는 의도다.

    # ⚠️ 함정: n_rows=8 에 값 종류가 극히 적다. 이렇게 작고 규칙적인 데이터
    # 위에서는 의미가 다른 두 질의가 우연히 같은 결과 집합을 내기 쉽다.
    # 즉 실행 동등성은 오수용(틀린 SQL 을 맞다고 받아들임) 쪽으로 편향된다.
    # 반대로 정답 SQL 자체가 빈 결과를 내면 예측도 빈 결과여야 통과하므로,
    # 이 지표는 절대 정확도가 아니라 **두 조건을 비교하기 위한 상대 지표**로만
    # 읽어야 한다.
    """
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)   # sqlite3 가 경로로 다시 열므로 여기서 fd 를 놓아 준다
    con = sqlite3.connect(path)
    try:
        for block in re.split(r"(?i)create\s+table", create_sql):
            block = block.strip()
            if not block:
                continue
            stmt = "CREATE TABLE " + block   # split 으로 떼어 낸 머리말을 되붙인다
            stmt = stmt.split(";")[0]        # 뒤에 딸려 온 다른 문장은 버린다
            try:
                con.execute(stmt)
            except Exception:
                continue   # 만들지 못한 테이블은 건너뛴다. 나머지는 계속 만든다
            m = re.match(r"(?i)create\s+table\s+[`\"]?(\w+)[`\"]?\s*\((.*)\)", stmt, flags=re.DOTALL)
            if not m:
                continue
            table = m.group(1)
            coldefs = [c.strip() for c in m.group(2).split(",") if c.strip()]
            cols, types = [], []
            for c in coldefs:
                cm = re.match(r"[`\"]?(\w+)[`\"]?\s+(\w+)", c)
                if cm:
                    cols.append(cm.group(1))
                    types.append(cm.group(2))   # 타입을 알아야 숫자/문자열을 맞춰 넣는다
            if not cols:
                continue
            ph = ",".join(["?"] * len(cols))   # 값은 항상 바인딩으로 넣는다
            for i in range(n_rows):
                # 열마다 i+j 로 옮겨 가며 값을 만든다. 모든 열이 같은 값으로
                # 채워지면 WHERE 절이 아무것도 걸러 내지 못한다.
                vals = [_rand_value(t, i + j) for j, t in enumerate(types)]
                try:
                    con.execute(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({ph})", vals)
                except Exception:
                    pass   # NOT NULL 등 제약에 걸린 행은 버리고 계속 채운다
        con.commit()
    finally:
        con.close()   # 예외가 나도 연결은 닫는다. 안 그러면 임시 파일이 잠긴다
    return path


def exec_equiv(create_sql: str, pred_sql: str, gold_sql: str) -> bool:
    """스키마로 만든 합성 데이터 DB 위에서 실행 동등성을 판정한다.

    호출마다 DB 를 새로 만들고 끝나면 지운다. 앞 문제의 테이블이 남아 있으면
    없는 테이블을 참조한 SQL 이 우연히 도는 일이 생기기 때문이다.
    """
    path = build_db(create_sql)
    try:
        # eval.sql_exec_match 는 행 순서를 무시하고 집합으로 비교한다.
        # 실행 자체가 실패하면 match 키가 없을 수 있어 get 으로 받는다.
        return bool(ev.sql_exec_match(path, pred_sql, gold_sql).get("match"))
    finally:
        try:
            os.remove(path)
        except Exception:
            pass   # 임시 파일 정리 실패가 채점 결과를 망치게 두지 않는다
