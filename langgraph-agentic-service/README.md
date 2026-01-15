# LangGraph Agentic Service

> **Multi-Agent RAG 시스템** - LangGraph 기반 Supervisor Pattern + Hybrid RAG + SSE Streaming

스마트팜 도메인을 위한 프로덕션급 Multi-Agent 시스템입니다. LangGraph StateGraph, Hybrid RAG(Dense+Sparse), FastAPI SSE Streaming, Redis 세션 관리, Langfuse 모니터링, RAGAS 평가를 통합한 End-to-End 솔루션입니다.

## 주요 기능

| 기능 | 설명 |
|------|------|
| **LangGraph Multi-Agent** | Supervisor Pattern으로 RAG/Tool/Conversation Agent 오케스트레이션 |
| **Hybrid RAG** | BGE-M3 (Dense + Sparse Embedding) + RRF Fusion |
| **Cross-Encoder Reranker** | bge-reranker-v2-m3 기반 정밀 재정렬 |
| **SSE Streaming** | Server-Sent Events 기반 실시간 스트리밍 응답 |
| **Redis Session** | Multi-turn 대화 컨텍스트 관리 |
| **Langfuse Monitoring** | 토큰 사용량, 지연시간, 품질 추적 |
| **Multi-tenant LoRA** | vLLM 동적 LoRA 어댑터 로딩 |
| **RAGAS Evaluation** | Faithfulness, Relevancy, Precision, Recall 평가 |

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI + SSE                            │
│  /chat/stream (SSE) │ /agent/execute │ /health                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph StateGraph                         │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────────┐   │
│  │Supervisor│───▶│  RAG Agent  │───▶│ Conversation Agent   │   │
│  │  Agent   │───▶│ Tool Agent  │    │                      │   │
│  └──────────┘    └─────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
         │                  │                      │
         ▼                  ▼                      ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────────────┐
│   Redis     │    │ Hybrid RAG   │    │      vLLM            │
│   Session   │    │ BGE-M3+RRF   │    │  Multi-LoRA          │
└─────────────┘    └──────────────┘    └──────────────────────┘
         │                  │                      │
         ▼                  ▼                      ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────────────┐
│  Langfuse   │    │  ChromaDB    │    │   HuggingFace        │
│  Monitoring │    │ Vector Store │    │   Model Hub          │
└─────────────┘    └──────────────┘    └──────────────────────┘
```

## 빠른 시작

### 1. 기본 실행 (CPU 모드)

```bash
# 저장소 클론
git clone <repo-url>
cd langgraph-agentic-service

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# Redis 실행 (Docker)
docker run -d --name redis -p 6379:6379 redis:7-alpine

# 서버 실행
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Docker Compose 실행

```bash
# 기본 서비스 (API + Redis + ChromaDB)
docker-compose up -d

# 개발 도구 포함
docker-compose --profile dev up -d

# 모니터링 포함 (Langfuse)
docker-compose --profile monitoring up -d

# GPU 서버 포함 (vLLM)
docker-compose --profile gpu up -d

# 모든 프로필
docker-compose --profile dev --profile monitoring --profile gpu up -d
```

### 3. API 테스트

```bash
# Health Check
curl http://localhost:8000/health

# SSE Streaming Chat
curl -N http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "토마토 재배 적정 온도는?",
    "session_id": "test-session-1"
  }'

# Multi-Agent Execute
curl http://localhost:8000/agent/execute \
  -H "Content-Type: application/json" \
  -d '{
    "message": "서울 날씨 알려줘",
    "session_id": "test-session-1",
    "tenant_id": "korea"
  }'
```

## 프로젝트 구조

```
langgraph-agentic-service/
├── configs/                    # YAML 설정 파일
│   ├── agent_config.yaml       # Agent 설정
│   ├── rag_config.yaml         # RAG 파이프라인 설정
│   ├── redis_config.yaml       # Redis 설정
│   └── langfuse_config.yaml    # Langfuse 설정
├── src/
│   ├── agents/                 # LangGraph Agents
│   │   ├── state.py            # AgentState 정의
│   │   ├── supervisor.py       # Supervisor Agent
│   │   ├── rag_agent.py        # RAG Agent
│   │   ├── tool_agent.py       # Tool Agent
│   │   └── conversation_agent.py
│   ├── rag/                    # Hybrid RAG Pipeline
│   │   ├── embedder.py         # BGE-M3 Embedder
│   │   ├── vector_store.py     # ChromaDB Store
│   │   ├── retriever.py        # Hybrid Retriever (RRF)
│   │   ├── reranker.py         # Cross-Encoder Reranker
│   │   └── pipeline.py         # RAG Pipeline
│   ├── api/                    # FastAPI Endpoints
│   │   ├── main.py             # App Entry
│   │   ├── routers/            # API Routers
│   │   └── schemas/            # Pydantic Models
│   ├── session/                # Redis Session
│   │   ├── redis_store.py      # Session Store
│   │   └── conversation_memory.py
│   ├── monitoring/             # Langfuse Integration
│   │   └── langfuse_tracker.py
│   ├── inference/              # LLM Inference
│   │   ├── vllm_client.py      # vLLM Client
│   │   └── lora_manager.py     # Multi-tenant LoRA
│   ├── tools/                  # Tool Implementations
│   │   ├── base_tool.py
│   │   ├── weather_tool.py
│   │   ├── yield_predictor.py
│   │   ├── temperature_predictor.py
│   │   └── panel_optimizer.py
│   ├── evaluation/             # RAGAS Evaluation
│   │   ├── ragas_evaluator.py
│   │   └── test_queries.json
│   └── services.py             # Service Layer
├── data/                       # 데이터 디렉토리
├── tests/                      # 테스트 코드
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## 핵심 컴포넌트

### 1. LangGraph Supervisor Agent

```python
from langgraph.graph import StateGraph, END
from src.agents.state import AgentState

# StateGraph 생성
graph = StateGraph(AgentState)

# 노드 추가
graph.add_node("supervisor", supervisor_agent)
graph.add_node("rag_agent", rag_agent)
graph.add_node("tool_agent", tool_agent)
graph.add_node("conversation_agent", conversation_agent)

# 조건부 엣지 (Supervisor가 다음 Agent 결정)
graph.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],
    {
        "rag_agent": "rag_agent",
        "tool_agent": "tool_agent",
        "conversation_agent": "conversation_agent",
        "END": END,
    }
)
```

### 2. Hybrid RAG (Dense + Sparse)

```python
from FlagEmbedding import BGEM3FlagModel

# BGE-M3로 Dense + Sparse 임베딩 동시 생성
model = BGEM3FlagModel("BAAI/bge-m3")
embeddings = model.encode(
    texts,
    return_dense=True,
    return_sparse=True,
)

# RRF (Reciprocal Rank Fusion)
def rrf_fusion(dense_results, sparse_results, k=60):
    scores = {}
    for rank, doc in enumerate(dense_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    for rank, doc in enumerate(sparse_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 3. SSE Streaming

```python
from sse_starlette.sse import EventSourceResponse

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        async for chunk in agent_service.execute_stream(request):
            yield {
                "event": chunk["type"],
                "data": json.dumps(chunk["data"])
            }
    return EventSourceResponse(event_generator())
```

### 4. RAGAS Evaluation

```python
from src.evaluation import RAGASEvaluator

evaluator = RAGASEvaluator()
result = await evaluator.evaluate(
    question="토마토 적정 온도는?",
    answer="토마토는 25-30도에서 잘 자랍니다.",
    contexts=retrieved_contexts,
    ground_truth="토마토 적정 온도: 25-30도"
)

# 메트릭
print(f"Faithfulness: {result.faithfulness}")
print(f"Answer Relevancy: {result.answer_relevancy}")
print(f"Context Precision: {result.context_precision}")
print(f"Context Recall: {result.context_recall}")
```

## 설정

### 환경 변수

```bash
# .env 파일
REDIS_HOST=localhost
REDIS_PORT=6379
VLLM_API_URL=http://localhost:8080
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
HF_TOKEN=hf_xxx
```

### RAG 설정 (configs/rag_config.yaml)

```yaml
embedder:
  model_name: "BAAI/bge-m3"
  max_length: 8192

retriever:
  hybrid_weights:
    dense: 0.7
    sparse: 0.3
  top_k: 10

reranker:
  model_name: "BAAI/bge-reranker-v2-m3"
  top_k: 5
```

## API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/chat/stream` | SSE 스트리밍 채팅 |
| POST | `/agent/execute` | Multi-Agent 실행 |
| GET | `/health` | 헬스 체크 |
| GET | `/health/ready` | Readiness Probe |
| GET | `/health/live` | Liveness Probe |

## 평가 실행

```bash
# RAGAS 평가 실행
python -m src.evaluation.ragas_evaluator

# 특정 테스트 쿼리로 평가
python -c "
import asyncio
from src.evaluation import RAGASEvaluator

async def main():
    evaluator = RAGASEvaluator()
    result = await evaluator.evaluate(
        question='토마토 재배 온도는?',
        answer='25-30도가 적합합니다.',
        contexts=['토마토 적정 온도는 25-30도입니다.']
    )
    print(result.to_dict())

asyncio.run(main())
"
```

## 기술 스택

| 분류 | 기술 |
|------|------|
| **Framework** | FastAPI, LangGraph, LangChain |
| **Embedding** | BGE-M3 (BAAI/bge-m3) |
| **Reranker** | bge-reranker-v2-m3 |
| **Vector DB** | ChromaDB |
| **Session** | Redis |
| **Monitoring** | Langfuse |
| **LLM Serving** | vLLM |
| **Evaluation** | RAGAS |

## 라이선스

MIT License
