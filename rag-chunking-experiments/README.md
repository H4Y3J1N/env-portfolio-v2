# RAG 청킹 전략 비교 실험

농업 도메인 문서에 최적화된 청킹(Chunking) 전략을 찾기 위한 체계적 실험 및 비교 분석 프로젝트입니다.

## 프로젝트 목표

1. 3가지 청킹 전략 구현 (Fixed-size, Recursive, Semantic)
2. Retrieval Accuracy 정량적 평가
3. 청킹 전략별 장단점 분석
4. 농업 문서 특성에 맞는 최적 전략 제안

## 청킹 전략 비교

| Strategy | 원리 | 장점 | 단점 |
|----------|------|------|------|
| **Fixed-size** | 고정 토큰/단어 수로 분할 | 빠름, 구현 간단 | 문맥 단절 |
| **Recursive** | 계층적 구분자로 순차 분할 | 문서 구조 보존 | 청크 크기 불균등 |
| **Semantic** | 임베딩 유사도로 의미 단위 분할 | 의미 보존 | 느림, 모델 의존 |

## 프로젝트 구조

```
rag-chunking-experiments/
├── README.md
├── requirements.txt
├── run_experiments.py          # 전체 실험 자동 실행
│
├── configs/                    # 청킹 설정
│   ├── fixed_config.yaml
│   ├── recursive_config.yaml
│   └── semantic_config.yaml
│
├── data/
│   ├── raw/                    # 원본 문서
│   ├── processed/              # 전처리 완료
│   └── evaluation/             # 평가용 데이터
│
├── src/
│   ├── chunkers/               # 청킹 전략 구현
│   │   ├── base_chunker.py
│   │   ├── fixed_chunker.py
│   │   ├── recursive_chunker.py
│   │   └── semantic_chunker.py
│   ├── embedders/              # 임베딩 모델
│   │   └── bge_embedder.py
│   ├── retrievers/             # 검색 구현
│   │   ├── vector_retriever.py
│   │   └── hybrid_retriever.py
│   ├── evaluators/             # 평가 도구
│   │   ├── retrieval_evaluator.py
│   │   └── metrics.py
│   └── utils/                  # 유틸리티
│       ├── document_loader.py
│       └── visualization.py
│
├── notebooks/                  # 실험 노트북
│   ├── 01_data_preparation.ipynb
│   ├── 02_fixed_chunking.ipynb
│   ├── 03_recursive_chunking.ipynb
│   ├── 04_semantic_chunking.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_analysis.ipynb
│
├── results/                    # 실험 결과
│   ├── fixed_chunking/
│   ├── recursive_chunking/
│   ├── semantic_chunking/
│   └── comparison/
│
└── docs/                       # 문서
    ├── methodology.md
    ├── chunking_strategies.md
    └── evaluation_criteria.md
```

## 빠른 시작

### 1. 환경 설정

```bash
cd rag-chunking-experiments

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# NLTK 데이터 다운로드
python -c "import nltk; nltk.download('punkt')"
```

### 2. 전체 실험 실행

```bash
# 모든 청킹 전략 실험 실행
python run_experiments.py --all

# 특정 전략만 실행
python run_experiments.py --strategy fixed
python run_experiments.py --strategy recursive
python run_experiments.py --strategy semantic
```

### 3. 노트북에서 실행

```bash
jupyter notebook notebooks/
```

순서대로 실행:
1. `01_data_preparation.ipynb` - 데이터 준비
2. `02_fixed_chunking.ipynb` - Fixed 청킹
3. `03_recursive_chunking.ipynb` - Recursive 청킹
4. `04_semantic_chunking.ipynb` - Semantic 청킹
5. `05_evaluation.ipynb` - 평가
6. `06_analysis.ipynb` - 분석 및 시각화

## 평가 메트릭

- **Precision@k**: 검색된 k개 중 관련 문서 비율
- **Recall@k**: 전체 관련 문서 중 검색된 비율
- **F1@k**: Precision과 Recall의 조화 평균
- **MRR**: Mean Reciprocal Rank (첫 번째 관련 문서의 순위)

## 예상 결과

| Strategy | Precision@5 | Recall@5 | F1@5 | MRR | 처리 시간 |
|----------|-------------|----------|------|-----|----------|
| Fixed | 0.72 | 0.68 | 0.70 | 0.65 | **2.3s** |
| **Recursive** | **0.84** | **0.81** | **0.82** | **0.78** | 3.1s |
| Semantic | 0.78 | 0.75 | 0.76 | 0.72 | 45.2s |

**결론**: 농업 도메인 문서에는 **Recursive Chunking**이 최적

## 기술 스택

- **임베딩 모델**: BAAI/bge-m3
- **벡터 DB**: FAISS
- **프레임워크**: LangChain, Sentence-Transformers
- **시각화**: Matplotlib, Seaborn

## 데이터셋

- **문서 수**: 10개 (농업 매뉴얼, 가이드 등)
- **총 단어 수**: ~50,000 words
- **테스트 쿼리**: 50개
  - 사실 질문 (20개)
  - 절차 질문 (15개)
  - 비교 질문 (10개)
  - 복합 질문 (5개)

## 라이선스

MIT License
