# 실험 방법론

## 1. 실험 목적

농업 도메인 문서에 최적화된 청킹(Chunking) 전략을 찾기 위한 체계적 비교 실험

## 2. 비교 대상

### 2.1 Fixed-size Chunking
- **원리**: 고정된 토큰/단어 수로 분할
- **설정**: 500 words, 50 words overlap
- **장점**: 빠름, 균일한 크기
- **단점**: 문맥 단절 가능

### 2.2 Recursive Chunking
- **원리**: 계층적 구분자로 순차 분할
- **구분자 우선순위**: 섹션 > 문단 > 줄 > 문장 > 단어
- **장점**: 문서 구조 보존
- **단점**: 크기 불균등

### 2.3 Semantic Chunking
- **원리**: 문장 임베딩 유사도 기반 분할
- **임계값**: 유사도 0.70 이하에서 분할
- **장점**: 의미 단위 보존
- **단점**: 느림, 모델 의존

## 3. 평가 메트릭

### 3.1 Retrieval Accuracy
- **Precision@k**: 검색된 k개 중 관련 문서 비율
- **Recall@k**: 전체 관련 문서 중 검색된 비율
- **F1@k**: Precision과 Recall의 조화 평균
- **MRR**: Mean Reciprocal Rank

### 3.2 Chunk Quality
- **평균 크기**: 청크당 평균 단어 수
- **표준편차**: 크기 분산
- **균일성**: 1 - (표준편차 / 평균)

### 3.3 Processing Efficiency
- **처리 시간**: 전체 청킹 소요 시간
- **처리량**: 청크/초

## 4. 데이터셋

### 4.1 문서 구성
- **도메인**: 농업 (영농형 태양광, 작물 재배)
- **형식**: 매뉴얼, 가이드, FAQ
- **총량**: ~50,000 단어

### 4.2 테스트 쿼리
- **총 개수**: 50개
- **카테고리**: 사실(20), 절차(15), 비교(10), 복합(5)
- **난이도**: Easy(20), Medium(20), Hard(10)

## 5. 실험 절차

```
1. 문서 로드 및 전처리
   └─> cleaned_documents.json

2. 청킹 실행 (3가지 전략)
   ├─> fixed_chunking/chunks.json
   ├─> recursive_chunking/chunks.json
   └─> semantic_chunking/chunks.json

3. 임베딩 생성 (BGE-M3)
   └─> 768차원 벡터

4. FAISS 인덱싱
   └─> 코사인 유사도 기반

5. 테스트 쿼리 검색 (Top-5)
   └─> retrieval_results.json

6. 평가 메트릭 계산
   └─> metrics.json

7. 결과 비교 및 분석
   └─> comparison/final_report.md
```

## 6. 환경

- **Python**: 3.10+
- **임베딩 모델**: BAAI/bge-m3
- **벡터 DB**: FAISS
- **GPU**: 선택 (임베딩 가속용)
