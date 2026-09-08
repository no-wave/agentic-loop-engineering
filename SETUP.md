# 환경 구축

**핵심: 설정 파일을 만들지 않아도 된다.** 아래 둘 중 하나만 준비하면 나머지는 노트북이 알아서 판별한다.

---

## 경로 A — OpenAI API (GPU 불필요, 5분)

노트북·맥북에서 전 과정을 완주할 수 있는 경로다.

```bash
pip install -r requirements-api.txt
export OPENAI_API_KEY=sk-...
jupyter lab
```

끝이다. 노트북 첫 셀을 실행하면 다음이 자동으로 정해진다.

| 항목 | 자동 선택값 | 바꾸려면 |
|------|-------------|----------|
| 챗 모델 | `gpt-5.6-luna` (저비용 기본값) | `export OPENAI_MODEL=gpt-5.6-terra` |
| 임베딩 | `text-embedding-3-small` | `export OPENAI_EMBED_MODEL=text-embedding-3-large` |

### 비용

임베딩은 `.cache/emb/` 에 디스크 캐시되므로 같은 노트북을 다시 돌려도 재과금되지 않는다. 코스에서 가장 무거운 11번(코퍼스 4,000건)도 임베딩 비용은 첫 실행 1회뿐이다.

```
text-embedding-3-small : $0.02 / 1M 토큰
gpt-5.6-luna           : $0.20 입력 / $1.20 출력  (1M 토큰당)
```

각 노트북 상단 배너에 전량 실행 시 예상 비용이 표시된다. 비용을 더 줄이려면 `export OPENAI_MODEL=gpt-5-nano` 로 바꾼다.

---

## 경로 B — 로컬 vLLM (GPU 필요, 자체 호스팅)

모델을 자기 GPU 에 직접 서빙하려면 이 경로를 쓴다. 기준 환경은 단일 NVIDIA A100 80GB 이고, 서빙 모델은 `Qwen/Qwen3-Coder-30B-A3B-Instruct` 다.

```bash
# vllm 을 먼저 설치해 드라이버에 맞는 torch 를 고르게 한다
uv pip install vllm --torch-backend=auto
pip install -r requirements.txt

# 서버 기동
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --port 8000

# 다른 터미널에서
jupyter lab
```

`localhost:8000` 이 응답하면 노트북이 자동으로 이 서버를 쓴다. **API 키가 설정돼 있어도 로컬을 우선한다** — 사용자가 모르는 사이에 과금되는 상황을 막기 위해서다.

포트나 호스트가 다르면 이것만 지정한다.

```bash
export VLLM_BASE_URL=http://<host>:<port>/v1
```

모델명은 지정할 필요가 없다. 서버에 `/v1/models` 를 물어 실제 서빙 중인 모델을 그대로 가져온다.

---

## 자동 판별이 어떻게 동작하는가

노트북 첫 셀의 `runtime.bootstrap()` 이 하는 일이다.

```
LLM_PROVIDER 확인
├─ "vllm"   → 로컬 강제. 응답 없으면 에러 (조용히 API 로 넘어가지 않는다)
├─ "openai" → API 강제. 키 없으면 에러
└─ 미지정   → 자동
     1. localhost:8000/v1/models 에 1.5초 프로브
        └─ 응답 → vLLM 사용 + 서빙 모델 자동 채택
     2. OPENAI_API_KEY 있음 → OpenAI API
     3. 둘 다 없음 → 해결 방법을 안내하는 에러
```

임베딩은 **LLM 과 독립적으로** 판별한다.

| 상황 | 임베딩 |
|------|--------|
| CUDA 가용 | `BAAI/bge-large-en-v1.5` (GPU) |
| CUDA 없음 + API 키 있음 | `text-embedding-3-small` |
| CUDA 없음 + 키 없음 | `BAAI/bge-small-en-v1.5` (CPU, 자동 강등) |

LLM 은 API 로 쓰면서 임베딩만 로컬 GPU 로 돌리는 조합, 그 반대 조합이 모두 성립한다. 판별 결과는 노트북 상단 배너에 항상 표시되므로, 내 수치가 어느 환경에서 나왔는지 헷갈릴 일이 없다.

### 판별 결과 강제하기

```bash
export LLM_PROVIDER=openai      # 로컬이 떠 있어도 API 를 쓴다
export EMBED_PROVIDER=local     # 임베딩만 로컬로
export LEN_REPROBE=1            # 판별 캐시를 버리고 다시 탐지
```

강제 지정한 백엔드를 쓸 수 없으면 **에러를 낸다**. 조용히 다른 백엔드로 넘어가면 어떤 모델이 그 수치를 냈는지 추적할 수 없게 되기 때문이다.

---

## 외부 데이터

10·15·16번은 원본이 `/ephemeral/hf/...` 절대경로를 전제한다. 한국어판은 `LEN_DATA_ROOT`(기본 `./data`) 아래에서 찾는다.

```
data/
├── gitbugs/hbase/hbase_bugs.csv          # 10, 11번
├── gitbugs/hbase/hbase_bugs-combined.csv
├── satd/technical_debt_dataset.csv       # 15번
└── commits.json                          # 16번
```

파일이 없으면 해당 셀이 예외로 죽지 않고, 내려받는 방법을 안내한 뒤 소규모 합성 데이터로 진행한다.

17번 캡스톤의 SWE-bench 검증은 Docker 가 필요하다. Docker 가 없으면 해당 절을 건너뛰고 원본 실행 결과를 인용해 설명만 진행한다.

---

## 문제 해결

**`❌ 사용 가능한 LLM 백엔드를 찾지 못했다`**
경로 A 또는 B 중 하나를 완료한다. 에러 메시지에 정확한 명령이 적혀 있다.

**임베딩이 너무 느리다**
CPU 로 동작 중일 가능성이 높다. 배너의 임베딩 줄을 확인한다. `OPENAI_API_KEY` 를 설정하면 API 임베딩으로 자동 전환된다.

**수치가 README 와 다르다**
API 모델은 원본(Qwen2.5-Coder-32B)과 다른 모델이므로 값이 다른 것이 정상이다. 원본 실측치는 `results/vllm/`, API 재실행 결과는 `results/openai/` 에 분리 보관한다.

**설정을 바꿨는데 반영되지 않는다**
`.len-runtime.json` 캐시 때문이다. `export LEN_REPROBE=1` 을 주거나 그 파일을 지운다.
