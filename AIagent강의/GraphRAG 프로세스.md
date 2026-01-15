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

### Retriever 유형별 비교

GraphRAG에서는 검색 방식에 따라 3가지 Retriever를 사용할 수 있습니다.

| Retriever | 검색 방식 | 특징 | 적합한 경우 |
|-----------|-----------|------|-------------|
| **VectorRetriever** | 벡터 유사도 검색 | 의미 기반 검색 | 비정형 텍스트에서 유사 문서 찾기 |
| **Text2CypherRetriever** | 자연어 → Cypher 변환 | LLM이 쿼리 생성 | 구조화된 데이터 질의 (집계, 필터링 등) |
| **VectorCypherRetriever** | 벡터 검색 + 그래프 순회 | 두 방식 조합 | 유사 노드 찾고 관계 정보까지 수집 |

> 각 Retriever의 상세 구현은 아래 섹션에서 설명합니다.

---

## Graph 기반 RAG 구현 (Text2Cypher)

> 참고: `250113 06.Graph 기반 GraphRAG실습.ipynb`

### 개념

Text2Cypher는 **자연어 질문을 Cypher 쿼리로 자동 변환**하는 방식입니다.

```
사용자 질문 (자연어)
        │
        ▼ LLM + DB 스키마
        │
Cypher 쿼리 자동 생성
        │
        ▼ Neo4j 실행
        │
검색 결과 반환
```

### 스키마 추출 (get_schema 함수)

LLM이 올바른 Cypher를 생성하려면 **DB 스키마 정보**가 필요합니다.

```python
from collections import defaultdict

def get_schema():
    schema = ""
    with driver.session() as session:
        # 모든 노드 라벨과 속성 추출
        node_schema = session.run("""
        CALL db.schema.nodeTypeProperties() YIELD nodeType, propertyName, propertyTypes
        RETURN nodeType, propertyName, propertyTypes
        """)

        nodes = defaultdict(dict)
        for record in node_schema:
            label = record["nodeType"].replace(":", "")
            prop = record["propertyName"]
            types = record["propertyTypes"]
            nodes[label][prop] = types[0] if types else "UNKNOWN"

        # 모든 관계 타입과 속성 추출
        rel_schema = session.run("""
        CALL db.schema.relTypeProperties() YIELD relType, propertyName, propertyTypes
        RETURN relType, propertyName, propertyTypes
        """)

        relationships = defaultdict(dict)
        for record in rel_schema:
            rel = record["relType"]
            prop = record["propertyName"]
            types = record["propertyTypes"]
            relationships[rel][prop] = types[0] if types else "UNKNOWN"

        # 관계 방향 및 타입 추출
        rel_types = session.run("""
        MATCH (a)-[r]->(b)
        RETURN DISTINCT labels(a) AS from_labels, type(r) AS rel_type, labels(b) AS to_labels
        """)

        rel_directions = set()
        for record in rel_types:
            from_label = f":{record['from_labels'][0]}"
            to_label = f":{record['to_labels'][0]}"
            rel_type = record['rel_type']
            rel_directions.add(f"({from_label})-[:{rel_type}]->({to_label})")

    # 스키마 문자열 생성
    schema += "\nNode properties:\n"
    for label, props in nodes.items():
        prop_str = ", ".join(f"{k}: {v}" for k, v in props.items())
        schema += f"{label} {{{prop_str}}}\n"

    schema += "\nRelationship properties:\n"
    for rel, props in relationships.items():
        prop_str = ", ".join(f"{k}: {v}" for k, v in props.items())
        schema += f"{rel} {{{prop_str}}}\n"

    schema += "\nThe relationships:\n"
    for rel in sorted(rel_directions):
        schema += f"{rel}\n"
    return schema
```

#### 스키마 출력 예시

```
Node properties:
Person {name: String, age: String}
Crime {id: String, date: String, type: String}
Location {address: String, postcode: String}

Relationship properties:
PARTY_TO {}
OCCURRED_AT {}

The relationships:
(:Crime)-[:OCCURRED_AT]->(:Location)
(:Person)-[:PARTY_TO]->(:Crime)
```

### Text2CypherRetriever 사용

```python
from neo4j_graphrag.retrievers import Text2CypherRetriever

# Few-shot 예시로 쿼리 품질 향상
examples = [
    """
    USER INPUT: Piccadilly에서 발생한 범죄 건수는?
    QUERY: MATCH (c:Crime)-[:OCCURRED_AT]->(l:Location {address: 'Piccadilly'})
    RETURN count(c) AS crime_count
    """
]

retriever = Text2CypherRetriever(
    driver=driver,
    llm=llm,
    neo4j_schema=get_schema(),
    examples=examples  # Few-shot 프롬프트
)

result = retriever.search(query_text="현재 수사중인 범죄 건수가 많은 담당자는?")
```

### GraphRAG 파이프라인 연결

```python
from neo4j_graphrag.generation import GraphRAG

graph_rag = GraphRAG(retriever, llm)

response = graph_rag.search(
    query_text="범죄자는 아니지만, 범죄자를 많이 알고 있는 사람은?",
    return_context=True
)

print(response.answer)
print(response.retriever_result.metadata["cypher"])  # 생성된 Cypher 확인
```

---

## Vector+Graph RAG 구현

> 참고: `250114 07.Vector+Graph RAG 기반 GraphRAG실습.ipynb`

### 개념

Vector+Graph RAG는 **벡터 검색으로 관련 노드를 찾고**, **그래프 순회로 추가 컨텍스트를 수집**하는 방식입니다.

```
사용자 질문
        │
        ▼ 임베딩 변환
        │
벡터 유사도 검색 (영화 줄거리로 영화 찾기)
        │
        ▼ 검색된 노드 기준
        │
그래프 순회 (배우, 감독, 장르 등 관계 정보 수집)
        │
        ▼
검색 결과 반환
```

### 사전 준비: 임베딩 추가

기존 노드에 임베딩이 없는 경우 추가합니다.

```python
from neo4j_graphrag.embeddings.sentence_transformers import SentenceTransformerEmbeddings

embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")  # 384차원

with driver.session() as session:
    result = session.run(
        "MATCH (m:Movie) WHERE m.plot IS NOT NULL RETURN elementId(m) AS id, m.plot AS plot"
    )
    records = result.data()

    for record in records:
        node_id = record["id"]
        text = record["plot"]
        vector = embedder.embed_query(text)

        session.run("""
        MATCH (m) WHERE elementId(m) = $id
        SET m.embedding = $embedding
        """, {"id": node_id, "embedding": vector})
```

### 사전 준비: 벡터 인덱스 생성

```python
from neo4j_graphrag.indexes import create_vector_index

create_vector_index(
    driver,
    name="plotindex",
    label="Movie",
    embedding_property="embedding",
    dimensions=384,
    similarity_fn="cosine"
)
```

### VectorCypherRetriever 사용

```python
from neo4j_graphrag.retrievers import VectorCypherRetriever

# 검색된 노드 기준 그래프 순회 쿼리
retrieval_query = """
MATCH (actor:Actor)-[:ACTED_IN]->(node)
RETURN
    node.title AS movie_title,
    node.plot AS movie_plot,
    collect(actor.name) AS actors
"""

retriever = VectorCypherRetriever(
    driver,
    index_name="plotindex",
    retrieval_query=retrieval_query,
    embedder=embedder
)

result = retriever.search(
    query_text="A movie about playing a board game in the jungle",
    top_k=5
)
```

### 한글 쿼리 주의사항

`all-MiniLM-L6-v2` 모델은 **영어 중심**으로 학습되어 한글 검색 성능이 낮습니다.

| 쿼리 | 예상 결과 |
|------|-----------|
| `"정글 보드게임 영화"` | ❌ 관련 없는 영화 반환 |
| `"A jungle board game movie"` | ✅ Jumanji 반환 |

**해결 방법**: GraphRAG 파이프라인에서 LLM이 한글을 이해하고 영어로 검색 후 한글로 응답합니다.

---

## GraphRAG Context 생성 과정

GraphRAG 파이프라인에서 `{context}` 변수가 어떻게 생성되는지 설명합니다.

### Context 생성 흐름

```
┌─────────────────────────────────────────────────────────────┐
│  graph_rag.search(query_text="정글 보드게임 영화 배우?")      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  1단계: Retriever 검색 (내부 자동 실행)                       │
│  retriever.search(query_text="...")                         │
│                                                             │
│  결과: [{movie_title: "Jumanji", actors: ["Robin Williams"]}]│
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  2단계: Context 자동 생성                                    │
│  검색 결과를 문자열로 변환                                    │
│                                                             │
│  context = "Movie: Jumanji\nActors: Robin Williams..."      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  3단계: 템플릿에 변수 삽입                                    │
│  Question: {query_text} → "정글 보드게임 영화 배우?"          │
│  Context: {context} → "Movie: Jumanji\nActors: ..."         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  4단계: LLM 응답 생성                                        │
└─────────────────────────────────────────────────────────────┘
```

### RagTemplate 사용

커스텀 프롬프트 템플릿을 정의할 수 있습니다.

```python
from neo4j_graphrag.generation import RagTemplate, GraphRAG

prompt_template = RagTemplate(
    template="""
    You are a helpful movie assistant.

    Use the context to include:
    - the movie title
    - a brief plot summary
    - main actor(s)

    Answer in Korean.

    Question: {query_text}

    Context: {context}

    Answer:
    """,
    expected_inputs=["context", "query_text"]
)

graph_rag = GraphRAG(retriever, llm, prompt_template=prompt_template)
```

### RagTemplate 구조

| 속성 | 설명 |
|------|------|
| `template` | 프롬프트 문자열 (`{context}`, `{query_text}` 포함) |
| `expected_inputs` | 템플릿에서 사용할 변수 이름 목록 |

> **핵심**: `{context}`는 사용자가 제공하지 않습니다. GraphRAG가 Retriever 검색 결과를 **자동으로 문자열 변환**하여 삽입합니다.

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

## 핵심 코드 상세 설명

### get_schema() 함수 분석

#### 1. defaultdict 사용 이유

```python
from collections import defaultdict

nodes = defaultdict(dict)
```

| 비교 | 일반 dict | defaultdict(dict) |
|------|-----------|-------------------|
| 없는 키 접근 | `KeyError` 발생 | 빈 딕셔너리 `{}` 자동 생성 |
| 코드 | `if key not in d: d[key] = {}` 필요 | 바로 `d[key][prop] = value` 가능 |

```python
# defaultdict 없이
nodes = {}
if label not in nodes:
    nodes[label] = {}
nodes[label][prop] = types[0]

# defaultdict 사용
nodes = defaultdict(dict)
nodes[label][prop] = types[0]  # 자동으로 nodes[label] = {} 생성
```

#### 2. 삼항 연산자 패턴

```python
nodes[label][prop] = types[0] if types else "UNKNOWN"
```

**구조**: `(참일 때 값) if (조건) else (거짓일 때 값)`

```python
# 풀어쓰면
if types:  # types 리스트가 비어있지 않으면
    nodes[label][prop] = types[0]  # 첫 번째 요소 사용
else:
    nodes[label][prop] = "UNKNOWN"  # 빈 리스트면 UNKNOWN
```

#### 3. 관계 방향 추출 코드

```python
rel_types = session.run("""
MATCH (a)-[r]->(b)
RETURN DISTINCT labels(a) AS from_labels, type(r) AS rel_type, labels(b) AS to_labels
""")

rel_directions = set()
for record in rel_types:
    from_label = f":{record['from_labels'][0]}"
    to_label = f":{record['to_labels'][0]}"
    rel_type = record['rel_type']
    rel_directions.add(f"({from_label})-[:{rel_type}]->({to_label})")
```

| 코드 | 설명 |
|------|------|
| `MATCH (a)-[r]->(b)` | 모든 관계 패턴 찾기 |
| `labels(a)` | 노드 a의 라벨 리스트 (예: `["Person"]`) |
| `labels(a)[0]` | 첫 번째 라벨 (예: `"Person"`) |
| `type(r)` | 관계 타입 (예: `"KNOWS"`) |
| `set()` | 중복 자동 제거 |

**결과 예시**: `"(:Person)-[:KNOWS]->(:Person)"`

#### 4. 스키마 문자열 생성

```python
schema += f"{label} {{{prop_str}}}\n"
```

| 표현 | 의미 |
|------|------|
| `{변수}` | 변수 값 삽입 |
| `{{` | 실제 `{` 문자 출력 |
| `}}` | 실제 `}` 문자 출력 |

```python
label = "Person"
prop_str = "name: String, age: Integer"

f"{label} {{{prop_str}}}"
# 결과: "Person {name: String, age: Integer}"
```

#### 5. .items()와 .join() 메서드

```python
for label, props in nodes.items():
    prop_str = ", ".join(f"{k}: {v}" for k, v in props.items())
```

| 메서드 | 설명 | 예시 |
|--------|------|------|
| `dict.items()` | (키, 값) 쌍의 리스트 반환 | `{"a": 1, "b": 2}.items()` → `[("a", 1), ("b", 2)]` |
| `", ".join(list)` | 리스트를 쉼표로 연결 | `", ".join(["a", "b"])` → `"a, b"` |

```python
props = {"name": "String", "age": "Integer"}
props.items()  # [("name", "String"), ("age", "Integer")]

# 제너레이터 표현식
f"{k}: {v}" for k, v in props.items()  # "name: String", "age: Integer"

# join으로 연결
", ".join(...)  # "name: String, age: Integer"
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
| **Text2Cypher** | 텍스트→사이퍼 | 자연어를 Cypher 쿼리로 변환 |
| **VectorCypherRetriever** | 벡터사이퍼검색기 | 벡터 검색 + 그래프 순회 조합 검색기 |
| **RagTemplate** | RAG 템플릿 | GraphRAG 프롬프트 템플릿 정의 클래스 |
| **Schema** | 스키마 | DB의 구조 정보 (노드, 관계, 속성) |
| **Sandbox** | 샌드박스 | 무료 클라우드 테스트 환경 |
| **Context** | 컨텍스트 | LLM에게 제공되는 검색 결과 문자열 |

---

## Neo4j 샌드박스 가이드

### 샌드박스란?

**샌드박스(Sandbox) = 무료 클라우드 테스트 환경**

| 특징 | 설명 |
|------|------|
| **무료** | 비용 없이 사용 가능 |
| **기간** | 3일간 유효 (연장 가능) |
| **설치 불필요** | 클라우드에서 실행, 인터넷만 있으면 접속 |
| **샘플 데이터** | Recommendations 등 데이터셋 미리 로드 가능 |

### 샌드박스 생성 방법

1. https://sandbox.neo4j.com 접속
2. 회원가입 / 로그인 (Google 계정 가능)
3. **"Create Sandbox"** 클릭
4. 원하는 데이터셋 선택 (예: **Recommendations**)
5. 생성 완료 후 연결 정보 확인

### 연결 정보 확인

**"Connection details"** 탭에서 확인:

| 항목 | 예시 |
|------|------|
| **Bolt URL** | `bolt://54.209.48.102:7687` |
| **Username** | `neo4j` |
| **Password** | `baby-grain-challenge` (자동 생성) |

### Python 연결 코드

```python
from neo4j import GraphDatabase, basic_auth

driver = GraphDatabase.driver(
    "neo4j://[IP주소]:7687",
    auth=basic_auth("neo4j", "[비밀번호]")
)

# 연결 테스트
with driver.session() as session:
    result = session.run("RETURN 1 AS test")
    print(result.single()["test"])  # 1 출력되면 성공
```

### 개인 샌드박스 vs 공유 서버

| 항목 | 개인 샌드박스 | 공유 서버 |
|------|---------------|-----------|
| **소유권** | 본인 계정에 귀속 | 여러 사용자 공용 |
| **데이터 격리** | 다른 사람 영향 없음 | 데이터 충돌 가능 |
| **수정/삭제** | 자유롭게 가능 | 다른 사람 데이터 영향 |

> **참고**: Neo4j Sandbox에서 생성하면 **개인 전용** 샌드박스입니다. "Connect via drivers"에 표시되는 연결 정보는 본인 샌드박스 정보입니다.

### 임베딩/인덱스 확인 쿼리

```cypher
-- 임베딩이 있는 노드 수 확인
MATCH (m:Movie) WHERE m.embedding IS NOT NULL
RETURN count(m) AS count

-- 벡터 인덱스 확인
SHOW INDEXES

-- 특정 인덱스 상세 확인
SHOW INDEXES YIELD name, type, state, populationPercent
WHERE name = 'plotindex'
```
