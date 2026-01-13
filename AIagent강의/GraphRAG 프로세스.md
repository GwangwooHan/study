# GraphRAG 전체 프로세스

GraphRAG는 크게 **2단계**로 나뉩니다: **인덱싱(구축)** 단계와 **검색(쿼리)** 단계입니다.

---

## Phase 1: Indexing (인덱싱 / 지식 그래프 구축)

데이터를 검색 가능한 형태로 준비하는 단계입니다.

```
┌────────────────────────────────────────────────────┐
│  INDEXING PHASE (오프라인 / 사전 준비)               │
├────────────────────────────────────────────────────┤
│                                                    │
│  ┌──────────────┐                                  │
│  │ 1. Source    │  PDF/텍스트 문서                  │
│  │    Document  │                                  │
│  └──────┬───────┘                                  │
│         │                                          │
│         ▼                                          │
│  ┌──────────────┐                                  │
│  │ 2. Chunking  │  텍스트 분할 (chunk_size=4000)   │
│  └──────┬───────┘                                  │
│         │                                          │
│         ▼                                          │
│  ┌──────────────┐                                  │
│  │ 3. Embedding │  벡터 변환 (1536차원)            │
│  └──────┬───────┘                                  │
│         │                                          │
│         ▼                                          │
│  ┌──────────────┐  LLM(GPT-4o)으로 추출            │
│  │ 4. Entity/   │  - Nodes (개체)                  │
│  │   Relation   │  - Relationships (관계)          │
│  │  Extraction  │  - FROM_CHUNK (연결)             │
│  └──────┬───────┘                                  │
│         │                                          │
│         ▼                                          │
│  ┌──────────────┐  Neo4j에 저장                    │
│  │ 5. Store     │  - Chunk 노드 (text, embedding)  │
│  │   (Neo4j)    │  - Entity 노드                   │
│  └──────┬───────┘  - 관계 (WON, MARRIED_TO 등)     │
│         │                                          │
│         ▼                                          │
│  ┌──────────────┐                                  │
│  │ 6. Vector    │  빠른 유사도 검색용 인덱스 생성   │
│  │    Index     │  (cosine similarity)             │
│  └──────────────┘                                  │
│                                                    │
└────────────────────────────────────────────────────┘
```

### 각 단계 설명

| 단계 | 용어 | 목적 | 노트북 코드 예시 |
|------|------|------|-----------------|
| **1** | **Chunking (청킹)** | 긴 문서를 검색 가능한 작은 단위로 분할 | `splitter.run(text=text)` |
| **2** | **Embedding (임베딩)** | 텍스트를 숫자 벡터로 변환 (의미 표현) | `chunk_embedder.run(text_chunks=chunks)` |
| **3** | **Entity/Relation Extraction (개체/관계 추출)** | LLM으로 텍스트에서 노드와 관계 추출 | `extractor.run(chunks=chunks)` |
| **4** | **Knowledge Graph Storage (지식 그래프 저장)** | 추출된 데이터를 Neo4j에 저장 | `writer.run(graph)` |
| **5** | **Vector Indexing (벡터 인덱싱)** | 빠른 유사도 검색을 위한 인덱스 생성 | `create_vector_index(...)` |

---

### 상세 설명

#### 1. Chunking (청킹)
```python
splitter = LangChainTextSplitterAdapter(
    CharacterTextSplitter(chunk_size=4000, chunk_overlap=200, separator=".")
)
chunks = await splitter.run(text=text)
```
- **목적**: LLM 컨텍스트 윈도우 제한 대응, 검색 정밀도 향상
- **결과**: `TextChunk` 객체 리스트

#### 2. Embedding (임베딩)
```python
embedder = OpenAIEmbeddings()  # text-embedding-ada-002 (1536차원)
chunks_with_embeddings = await chunk_embedder.run(text_chunks=chunks)
```
- **목적**: 텍스트의 **의미적 유사성**을 수치로 표현
- **결과**: 각 Chunk에 `[0.12, -0.34, ...]` 형태의 벡터 추가

> **중요: 임베딩 단위는 "청크 전체"이며, 단어별 임베딩이 아닙니다.**

##### 임베딩 단위 설명 (예시)

```
원본 문서:
"Marie Curie was a Polish physicist. She won two Nobel Prizes.
Her husband Pierre Curie was also a scientist."
```

**Chunking 후:**
```
Chunk 1: "Marie Curie was a Polish physicist."
Chunk 2: "She won two Nobel Prizes."
Chunk 3: "Her husband Pierre Curie was also a scientist."
```

**Embedding 후:**
```
Chunk 1: "Marie Curie was a Polish physicist."
         → [0.12, -0.34, 0.56, ..., 0.23]  ← 1536차원 벡터 1개

Chunk 2: "She won two Nobel Prizes."
         → [0.08, 0.45, -0.12, ..., 0.67]  ← 1536차원 벡터 1개

Chunk 3: "Her husband Pierre Curie was also a scientist."
         → [-0.23, 0.11, 0.78, ..., -0.45] ← 1536차원 벡터 1개
```

##### 단어별 vs 청크별 임베딩 비교

| 방식 | 설명 | 문제점 |
|------|------|--------|
| **단어별 임베딩 (사용 안 함)** | "Marie" → 벡터, "Curie" → 벡터... | 문맥 손실, 검색 불가능 |
| **청크별 임베딩 (실제 방식)** | "Marie Curie was a Polish physicist." → 벡터 1개 | 문장 전체의 **의미**를 표현 |

##### 청크 단위 임베딩을 사용하는 이유

| 이유 | 설명 |
|------|------|
| **의미 검색** | "노벨상 수상자가 누구야?" 질문 시, 문장 단위로 검색해야 의미 있는 답변 가능 |
| **문맥 보존** | "She"가 누구인지는 문장 전체를 봐야 알 수 있음 |
| **효율성** | 단어별로 저장하면 벡터 수가 폭발적으로 증가 |

##### 검색 시 동작 예시

```
질문: "Who won Nobel Prize?"
         ↓
질문 임베딩: [0.07, 0.42, -0.15, ..., 0.61]
         ↓
코사인 유사도 비교:
  - Chunk 1 벡터와 유사도: 0.72
  - Chunk 2 벡터와 유사도: 0.94  ← 가장 유사! (Nobel Prize 언급)
  - Chunk 3 벡터와 유사도: 0.65
         ↓
Chunk 2 반환: "She won two Nobel Prizes."
```

#### 3. Entity/Relation Extraction (개체/관계 추출)
```python
extractor = LLMEntityRelationExtractor(llm=OpenAILLM(...))
extract_results = await extractor.run(chunks=chunks_with_embeddings)
```
- **목적**: 텍스트에서 **구조화된 지식** 추출
- **결과**:
  - **Nodes**: `Marie Curie`, `Nobel Prize`, `University of Paris`
  - **Relationships**: `Marie Curie -[WON]-> Nobel Prize`

#### 4. Knowledge Graph Storage (지식 그래프 저장)
```python
writer = Neo4jWriter(driver)
graph = Neo4jGraph(nodes=extract_results.nodes, relationships=extract_results.relationships)
await writer.run(graph)
```
- **목적**: 추출된 지식을 **그래프 DB**에 영구 저장
- **결과**: Neo4j에 노드와 관계 생성

#### 5. Vector Indexing (벡터 인덱싱)
```python
create_vector_index(
    driver, "vectorchunk",
    label="Chunk", embedding_property="embedding",
    dimensions=1536, similarity_fn="cosine"
)
```
- **목적**: 벡터 검색 속도 최적화 (O(n) → O(log n))
- **결과**: Neo4j 내부에 벡터 인덱스 생성

---

## Phase 2: Retrieval & Generation (검색 및 생성)

사용자 질문에 답변하는 단계입니다.

```
┌────────────────────────────────────────────────────┐
│  RETRIEVAL & GENERATION PHASE (온라인 / 실시간)     │
├────────────────────────────────────────────────────┤
│                                                    │
│  ┌──────────────┐                                  │
│  │ 1. Query     │  "Who is Marie Curie?"           │
│  │    Input     │                                  │
│  └──────┬───────┘                                  │
│         │                                          │
│         ▼                                          │
│  ┌──────────────┐                                  │
│  │ 2. Query     │  질문을 벡터로 변환 (1536차원)    │
│  │   Embedding  │  (동일한 임베딩 모델 사용)        │
│  └──────┬───────┘                                  │
│         │                                          │
│         ▼                                          │
│  ┌──────────────┐  코사인 유사도 기반 검색          │
│  │ 3. Vector    │  ┌─────────────────┐             │
│  │    Search    │  │ 유사도 순위      │             │
│  │              │  │ 1. Chunk A 0.94 │             │
│  │              │  │ 2. Chunk B 0.87 │             │
│  │              │  │ 3. Chunk C 0.82 │             │
│  └──────┬───────┘  └─────────────────┘             │
│         │                                          │
│         ▼                                          │
│  ┌──────────────┐                                  │
│  │ 4. Context   │  검색된 Chunk + 그래프 정보 조합  │
│  │   Assembly   │  (top_k개 문서 수집)              │
│  └──────┬───────┘                                  │
│         │                                          │
│         ▼                                          │
│  ┌──────────────┐                                  │
│  │ 5. LLM       │  GPT-4o 등 LLM으로 답변 생성     │
│  │  Generation  │                                  │
│  └──────┬───────┘                                  │
│         │                                          │
│         ▼                                          │
│  ┌──────────────┐  response 객체 반환              │
│  │ 6. Response  │  - answer: 최종 답변             │
│  │              │  - retriever_result: 검색 정보   │
│  └──────────────┘                                  │
│                                                    │
└────────────────────────────────────────────────────┘
```

### 각 단계 설명

| 단계 | 용어 | 목적 |
|------|------|------|
| **1** | **Query Embedding (쿼리 임베딩)** | 사용자 질문을 벡터로 변환 (동일한 임베딩 모델 사용) |
| **2** | **Vector Search (벡터 검색)** | 질문과 유사한 Chunk를 인덱스에서 코사인 유사도로 검색 |
| **3** | **Context Assembly (컨텍스트 조합)** | 검색된 Chunk(top_k개)와 연결된 그래프 정보 수집 |
| **4** | **LLM Generation (LLM 생성)** | 컨텍스트 기반으로 답변 생성 (GPT-4o 등) |
| **5** | **Response (응답 반환)** | `answer`(답변)와 `retriever_result`(검색 컨텍스트) 반환 |

### 코드 예시
```python
retriever = VectorRetriever(driver, "vectorchunk", embedder=OpenAIEmbeddings())
llm = OpenAILLM(model_name="gpt-4o")
graph_rag = GraphRAG(retriever, llm)

response = graph_rag.search(
    query_text="Who is Marie Curie?",
    retriever_config={"top_k": 3},
    return_context=True
)
```

### search() Arguments

| 파라미터 | 설명 |
|----------|------|
| `query_text` | 검색할 질문 문자열 |
| `retriever_config` | 검색기 설정 (`top_k`: 반환할 문서 수) |
| `return_context` | `True`면 `retriever_result`에 검색된 Chunk 정보 포함 |

---

### GraphRAG 응답(Response) 구조

`graph_rag.search()`의 반환값은 크게 **2개의 주요 속성**으로 구성됩니다.

```
response
├── answer                 # LLM이 생성한 최종 답변
└── retriever_result       # 검색된 컨텍스트 정보
    ├── items[]            # 검색된 문서(Chunk) 리스트
    │   ├── content        # 문서 내용 (embedding은 None으로 생략)
    │   └── metadata       # score, nodeLabels, id
    └── metadata
        └── query_vector   # 질문의 임베딩 벡터
```

#### 응답 접근 방법

```python
# 최종 답변
print(response.answer)

# 검색된 문서 개수
print(len(response.retriever_result.items))

# 첫 번째 문서의 유사도 점수
print(response.retriever_result.items[0].metadata['score'])

# 첫 번째 문서의 내용
print(response.retriever_result.items[0].content)

# 질문의 임베딩 벡터
print(response.retriever_result.metadata['query_vector'])
```

#### 응답 속성 요약

| 속성 | 타입 | 설명 |
|------|------|------|
| `response.answer` | `str` | LLM이 생성한 최종 답변 |
| `response.retriever_result.items` | `List` | 검색된 문서 리스트 |
| `items[n].content` | `str` | n번째 문서 내용 (JSON 형식) |
| `items[n].metadata['score']` | `float` | n번째 문서 유사도 점수 (0~1) |
| `metadata['query_vector']` | `List[float]` | 질문 임베딩 벡터 |

#### content 내 embedding이 None인 이유

| 이유 | 설명 |
|------|------|
| **성능 최적화** | 임베딩 벡터는 384~1536개 숫자로 매우 큼 |
| **응답 크기 절감** | 매번 반환하면 응답이 너무 커짐 |
| **내부 사용** | 검색 시 내부적으로만 사용, 결과에는 생략 |

> 실제 임베딩은 **Neo4j 노드의 `embedding` 속성**에 저장되어 있습니다.

---

## 전체 흐름 요약

### INDEXING (구축 단계)

```
Document ──▶ Chunking ──▶ Embedding ──▶ Neo4j Storage ──▶ Vector Index
                │                         │
                ▼                         ▼
              LLM ──▶ Nodes/Relations ──▶ Chunk 노드 + Entity 노드 + 관계
```

**저장되는 데이터:**
- **Chunk 노드**: text, embedding (벡터)
- **Entity 노드**: Marie Curie, Nobel Prize 등
- **관계**: FROM_CHUNK, WON, MARRIED_TO 등

### RETRIEVAL (검색 단계)

```
Query ──▶ Embedding ──▶ Vector Search ──▶ Context ──▶ LLM ──▶ Response
                            │                              │
                            ▼                              ▼
                     유사도 순위 (top_k)              answer + retriever_result
```

### Neo4j에 저장되는 구조

```
┌────────────────────────────────────────────────────┐
│  Neo4j Graph Database                              │
│                                                    │
│  ┌───────────┐                                     │
│  │  Chunk    │──FROM_CHUNK──▶┌─────────────┐      │
│  │ text,     │               │ Marie Curie │      │
│  │ embedding │               └──────┬──────┘      │
│  └───────────┘                      │             │
│                           ┌────────┬┴────────┐    │
│                          WON              MARRIED_TO│
│                           │                  │    │
│                           ▼                  ▼    │
│                    ┌────────────┐    ┌───────────┐│
│                    │Nobel Prize │    │Pierre Curie││
│                    └────────────┘    └───────────┘│
│                                                    │
│  Vector Index: Chunk.embedding (cosine)           │
└────────────────────────────────────────────────────┘
```

---

## 벡터 인덱스 확인 방법

벡터 인덱스는 그래프 시각화 화면에 보이지 않습니다. Neo4j Browser에서 확인:

```cypher
SHOW INDEXES
-- 또는
SHOW VECTOR INDEXES
```

---

## 임베딩 직접 추가하기

기존 노드에 임베딩이 없는 경우 수동으로 추가하는 방법입니다.

```python
from neo4j_graphrag.embeddings.sentence_transformers import SentenceTransformerEmbeddings

embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")  # 384차원
driver = GraphDatabase.driver(URI, auth=AUTH)

with driver.session() as session:
    # 1. 임베딩 없는 노드 조회
    result = session.run(
        "MATCH (q:Question) WHERE q.embedding IS NULL "
        "RETURN elementId(q) AS id, q.body_markdown AS text"
    )
    records = result.data()

    # 2. 각 노드에 임베딩 추가
    for record in records:
        node_id = record["id"]
        text = record["text"]
        vector = embedder.embed_query(text)  # 텍스트 → 벡터

        session.run("""
        MATCH (q) WHERE elementId(q) = $id
        SET q.embedding = $embedding
        """, {"id": node_id, "embedding": vector})
```

---

## SimpleKGPipeline 사용 시 주의사항

PDF에서 지식 그래프를 자동 구축할 때 JSON 파싱 오류가 발생할 수 있습니다.

### 오류 해결: response_format 지정

```python
from neo4j_graphrag.llm import OpenAILLM

# JSON 응답 형식 강제 지정
llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "response_format": {"type": "json_object"}  # 핵심!
    }
)

kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    embedder=OpenAIEmbeddings(),
    from_pdf=True
)

await kg_builder.run_async(file_path="GraphRAG.pdf")
```

---

## 용어 정리표

| 영어 용어 | 한글 용어 | 설명 |
|-----------|-----------|------|
| **Chunking** | 청킹 | 문서를 작은 단위로 분할 |
| **Embedding** | 임베딩 | 텍스트를 벡터로 변환 |
| **Entity Extraction** | 개체 추출 | 텍스트에서 명사/개념 추출 |
| **Relation Extraction** | 관계 추출 | 개체 간 관계 추출 |
| **Knowledge Graph** | 지식 그래프 | 노드와 관계로 구성된 그래프 |
| **Vector Index** | 벡터 인덱스 | 빠른 유사도 검색용 인덱스 |
| **Retriever** | 검색기 | 관련 문서를 찾는 컴포넌트 |
| **RAG** | 검색 증강 생성 | Retrieval-Augmented Generation |
| **GraphRAG** | 그래프 RAG | 그래프 구조를 활용한 RAG |
