# 청킹 전략 상세 설명

## 1. 청킹(Chunking)이란?

청킹은 긴 문서를 검색 가능한 작은 단위로 분할하는 과정입니다. RAG(Retrieval-Augmented Generation) 시스템에서 핵심적인 전처리 단계입니다.

### 왜 청킹이 중요한가?

1. **LLM 컨텍스트 제한**: 대부분의 LLM은 입력 토큰 수 제한이 있음
2. **검색 정밀도**: 적절한 크기의 청크가 더 정확한 검색 가능
3. **비용 효율**: 필요한 부분만 LLM에 전달하여 비용 절감

---

## 2. Fixed-size Chunking

### 원리

텍스트를 고정된 크기(단어 수, 문자 수, 토큰 수)로 기계적으로 분할

```python
def fixed_chunk(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks
```

### 예시

**원본:**
```
토마토 재배 시 주의사항은 다음과 같습니다. 첫째, 적절한 온도를
유지해야 합니다. 낮 기온은 25-28도가 적합합니다. 둘째, 충분한
일조량이 필요합니다. 하루 6시간 이상의 햇빛을 받아야 합니다.
```

**Fixed Chunking (20 words):**
```
Chunk 1: "토마토 재배 시 주의사항은 다음과 같습니다. 첫째, 적절한 온도를 유지해야 합니다. 낮 기온은 25-28도가 적합합니다."
Chunk 2: "적합합니다. 둘째, 충분한 일조량이 필요합니다. 하루 6시간 이상의 햇빛을 받아야 합니다."
```

### 장단점

| 장점 | 단점 |
|------|------|
| 구현이 간단함 | 문장 중간에서 분할 가능 |
| 처리 속도가 빠름 | 문맥이 끊길 수 있음 |
| 청크 크기가 균일함 | 의미 단위를 무시함 |

---

## 3. Recursive Chunking

### 원리

계층적 구분자를 사용하여 문서 구조를 보존하면서 분할

```python
separators = [
    "\n\n\n",  # 섹션
    "\n\n",    # 문단
    "\n",      # 줄
    ". ",      # 문장
    " "        # 단어
]

def recursive_chunk(text, separators, max_size):
    if len(text.split()) <= max_size:
        return [text]

    sep = separators[0]
    parts = text.split(sep)

    chunks = []
    current = ""

    for part in parts:
        if len((current + part).split()) <= max_size:
            current += part + sep
        else:
            if current:
                chunks.append(current)
            if len(part.split()) > max_size:
                # 재귀: 다음 구분자로
                chunks.extend(recursive_chunk(part, separators[1:], max_size))
            else:
                current = part + sep

    if current:
        chunks.append(current)

    return chunks
```

### 예시

**원본:**
```
# 토마토 재배

## 온도 관리
낮 기온: 25-28도
밤 기온: 15-18도

## 물 관리
아침에 관수
과습 주의
```

**Recursive Chunking:**
```
Chunk 1: "# 토마토 재배\n\n## 온도 관리\n낮 기온: 25-28도\n밤 기온: 15-18도"
Chunk 2: "## 물 관리\n아침에 관수\n과습 주의"
```

### 장단점

| 장점 | 단점 |
|------|------|
| 문서 구조 보존 | 청크 크기 불균등 |
| 자연스러운 분할점 | 구현이 복잡함 |
| 문맥 연속성 유지 | 최적 구분자 선택 필요 |

---

## 4. Semantic Chunking

### 원리

문장 임베딩 간 유사도를 계산하여 의미가 변하는 지점에서 분할

```python
def semantic_chunk(text, embedder, threshold=0.7):
    sentences = sent_tokenize(text)
    embeddings = embedder.encode(sentences)

    # 인접 문장 간 유사도
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i+1])
        similarities.append(sim)

    # 유사도가 threshold 미만인 곳에서 분할
    split_points = [0]
    for i, sim in enumerate(similarities):
        if sim < threshold:
            split_points.append(i + 1)

    # 청크 생성
    chunks = []
    for i in range(len(split_points) - 1):
        chunk = " ".join(sentences[split_points[i]:split_points[i+1]])
        chunks.append(chunk)

    return chunks
```

### 예시

**원본:**
```
토마토는 고온을 좋아합니다. 최적 온도는 25-28도입니다.
[유사도: 0.85 - 같은 주제]

물 관리도 중요합니다. 과습은 피해야 합니다.
[유사도: 0.45 - 주제 전환! 분할점]

적절한 배수가 필요합니다.
```

**Semantic Chunking:**
```
Chunk 1: "토마토는 고온을 좋아합니다. 최적 온도는 25-28도입니다."
Chunk 2: "물 관리도 중요합니다. 과습은 피해야 합니다. 적절한 배수가 필요합니다."
```

### 장단점

| 장점 | 단점 |
|------|------|
| 의미 단위 보존 | 처리 속도 느림 |
| 주제별 일관성 | 임베딩 모델 필요 |
| 검색 품질 향상 | 청크 크기 불균등 |

---

## 5. 전략 선택 가이드

| 상황 | 권장 전략 |
|------|----------|
| 실시간 처리 필요 | Fixed |
| 구조화된 문서 (매뉴얼, 가이드) | Recursive |
| 고품질 검색 필요 | Semantic |
| 대용량 데이터 | Fixed |
| 오프라인 인덱싱 가능 | Semantic |

---

## 6. 참고 자료

- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Semantic Chunking Paper](https://arxiv.org/abs/2301.00303)
- [RAG Best Practices](https://www.pinecone.io/learn/chunking-strategies/)
