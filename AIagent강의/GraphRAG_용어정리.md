# GraphRAG 용어 정리

> GraphRAG와 Cypher(Neo4j) 관련 핵심 용어를 체계적으로 정리한 문서

---

## 1. 핵심 용어 매핑표

| GraphRAG 용어 | Cypher/Neo4j 용어 | 설명 | 예시 |
|--------------|------------------|------|------|
| **Entity** | **Node** | 개체/객체 (그래프의 점) | Tom Hanks, Forrest Gump |
| **Entity Type** | **Node Label** | 개체의 종류/타입 | Person, Movie, Actor |
| **Relationship** | **Relationship** | 개체 간 연결 (그래프의 선) | ACTED_IN, DIRECTED |
| **Relationship Type** | **Relationship Type** | 관계의 종류 | :ACTED_IN, :KNOWS |
| **Attribute** | **Property** | 개체/관계의 속성값 | name, age, year |
| **Triple** | **Pattern** | (주어, 관계, 목적어) 구조 | (s, r, t) = (s)-[r]->(t) |
| **Knowledge Graph** | **Graph Database** | 전체 그래프 구조 | Neo4j DB |

---

## 2. 개념별 상세 설명

### 2.1 Entity / Node (개체 / 노드)

| 구분 | GraphRAG | Cypher |
|------|----------|--------|
| **명칭** | Entity (엔티티) | Node (노드) |
| **정의** | 텍스트에서 추출한 개체 | 그래프DB에 저장된 점 |
| **역할** | LLM이 인식한 고유 명사/개념 | 쿼리/저장 대상 |

```
GraphRAG 추출:  "Tom Hanks는 Forrest Gump에 출연했다"
                    ↓ Entity 추출
              Entity: Tom Hanks (Person)
              Entity: Forrest Gump (Movie)

Cypher 저장:   (tom:Person {name: "Tom Hanks"})
               (fg:Movie {title: "Forrest Gump"})
```

### 2.2 Entity Type / Node Label (개체 타입 / 노드 레이블)

| 구분 | GraphRAG | Cypher |
|------|----------|--------|
| **명칭** | Entity Type | Node Label |
| **정의** | 엔티티 분류 카테고리 | 노드에 붙는 태그 |
| **결정 주체** | LLM이 결정 | `:` 뒤에 표기 |
| **다중 지정** | 가능 | 가능 (`:Person:Actor`) |

```python
# GraphRAG 스키마 정의
entities = ["Person", "Movie", "Director"]  # Entity Types

# Cypher에서는 Node Label로 표현
CREATE (n:Person {name: "Tom Hanks"})      # 단일 레이블
CREATE (n:Person:Actor {name: "Tom Hanks"}) # 다중 레이블
```

### 2.3 Relationship (관계)

| 구분 | GraphRAG | Cypher |
|------|----------|--------|
| **표현** | (source, relation, target) | `()-[]-()` |
| **방향** | 튜플 순서로 암시 | `->`, `<-`, `--`로 명시 |
| **타입** | 관계 문자열 | `:RELATIONSHIP_TYPE` |

```
GraphRAG:  ("Tom Hanks", "ACTED_IN", "Forrest Gump")
              source      relation      target

Cypher:    (tom)-[:ACTED_IN]->(movie)
             ↑        ↑          ↑
           source    type      target
```

**Cypher 방향 표기:**
| 표기 | 의미 |
|------|------|
| `->` | 왼쪽에서 오른쪽으로 |
| `<-` | 오른쪽에서 왼쪽으로 |
| `--` | 방향 무관 (양방향 검색) |

### 2.4 Attribute / Property (속성)

| 구분 | GraphRAG | Cypher |
|------|----------|--------|
| **명칭** | Attribute | Property |
| **적용 대상** | Entity, Relationship | Node, Relationship |
| **표기** | 딕셔너리 형태 | `{key: value}` |

```cypher
-- Node Property
(p:Person {name: "Tom Hanks", age: 65})

-- Relationship Property
(p)-[:ACTED_IN {role: "Forrest", year: 1994}]->(m)
```

---

## 3. 전체 구조 비교도

```
┌─────────────────────────────────────────────────────────────┐
│                     GraphRAG 관점                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Entity Type: Person          Entity Type: Movie           │
│        ┌──────────┐               ┌──────────┐              │
│        │ Entity   │               │ Entity   │              │
│        │──────────│  Relationship │──────────│              │
│        │ Tom Hanks│──────────────→│ Forrest  │              │
│        │          │   ACTED_IN    │ Gump     │              │
│        │Attribute:│               │Attribute:│              │
│        │ age: 65  │               │ year:1994│              │
│        └──────────┘               └──────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘

                           ↓ 변환 ↓

┌─────────────────────────────────────────────────────────────┐
│                    Cypher/Neo4j 관점                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Label: Person                Label: Movie                 │
│        ┌──────────┐               ┌──────────┐              │
│        │  Node    │               │  Node    │              │
│        │──────────│  Relationship │──────────│              │
│        │(tom)     │──[:ACTED_IN]─→│(movie)   │              │
│        │          │      Type     │          │              │
│        │Property: │               │Property: │              │
│        │name:"Tom"│               │title:    │              │
│        │age: 65   │               │"Forrest" │              │
│        └──────────┘               └──────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Cypher 문법 요약

### 4.1 기본 문법 구조

```cypher
-- 전체 패턴 구조
MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
WHERE p.name = "Tom Hanks"
RETURN p, r, m

--     ↑         ↑            ↑
--   Node    Relationship   Node
--  :Person  :ACTED_IN     :Movie
--  (Label)    (Type)      (Label)
```

### 4.2 주요 절(Clause)

| 절 | 역할 | 예시 |
|----|------|------|
| `CREATE` | 노드/관계 생성 | `CREATE (n:Person {name: "Tom"})` |
| `MATCH` | 패턴 검색 | `MATCH (n:Person)` |
| `WHERE` | 조건 필터링 | `WHERE n.age > 30` |
| `RETURN` | 결과 반환 | `RETURN n.name` |
| `SET` | 속성 수정 | `SET n.age = 40` |
| `DELETE` | 노드/관계 삭제 | `DELETE n` |
| `MERGE` | 있으면 매칭, 없으면 생성 | `MERGE (n:Person {name: "Tom"})` |

### 4.3 속성 접근

```cypher
-- 노드 속성 접근
p.name        -- 노드 p의 name 속성
p["name"]     -- 동일한 표현

-- 관계 속성 접근
r.role        -- 관계 r의 role 속성
```

---

## 5. 실제 코드 흐름 예시

```python
# 1. GraphRAG: 스키마 정의 (Entity Types, Relationship Types)
allowed_nodes = ["Person", "Movie", "Director"]
allowed_relationships = ["ACTED_IN", "DIRECTED"]

# 2. GraphRAG: LLM으로 텍스트에서 추출
entities = [
    {"name": "Tom Hanks", "type": "Person"},      # Entity
    {"name": "Forrest Gump", "type": "Movie"}     # Entity
]
relationships = [
    ("Tom Hanks", "ACTED_IN", "Forrest Gump")     # Relationship (Triple)
]

# 3. Cypher: Neo4j에 저장
"""
MERGE (p:Person {name: "Tom Hanks"})      // Node with Label
MERGE (m:Movie {title: "Forrest Gump"})   // Node with Label
MERGE (p)-[:ACTED_IN]->(m)                // Relationship with Type
"""

# 4. Cypher: 질의
"""
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE p.name = "Tom Hanks"
RETURN m.title
"""
```

---

## 6. 빠른 참조 카드

```
GraphRAG        ↔        Cypher
────────────────────────────────
Entity          =        Node
Entity Type     =        Label
Attribute       =        Property
Relationship    =        Relationship
Relation Type   =        Relationship Type
Triple          =        Pattern
(s, r, t)       =        (s)-[r]->(t)
```

| 키워드 | 의미 |
|--------|------|
| `(n)` | 노드 (변수 n) |
| `(n:Label)` | Label 타입의 노드 |
| `{key: value}` | 속성 |
| `-[r]->` | 관계 (오른쪽 방향) |
| `[:TYPE]` | 관계 타입 |
| `MATCH` | 패턴 검색 |
| `MERGE` | 있으면 매칭, 없으면 생성 |

---

## 7. LLM 프롬프트 용어

### 7.1 No pre-amble (서문 없이)

LLM에게 **불필요한 도입부/서문 없이 바로 본론만** 응답하도록 지시하는 표현.

**preamble** = 서문, 전문, 도입부

```
❌ With preamble (서문 있음):
"Sure! I'd be happy to help you with that. Here's the Cypher query you requested:
MATCH (p:Person) RETURN p"

✅ No preamble (서문 없음):
"MATCH (p:Person) RETURN p"
```

### 7.2 자주 쓰는 LLM 지시 표현

| 표현 | 의미 | 용도 |
|------|------|------|
| `No pre-amble` | 서문/도입부 없이 | 바로 본론만 |
| `No explanations` | 설명 없이 | 결과만 출력 |
| `No apologies` | 사과 없이 | "죄송합니다만..." 금지 |
| `Respond with X only` | X만 응답해라 | 다른 텍스트 금지 |
| `Do not wrap in backticks` | 백틱으로 감싸지 마라 | 순수 코드만 |

### 7.3 왜 필요한가?

LLM은 기본적으로 친절하게 답하려고 이런 말을 붙임:
- "Sure, I can help with that!"
- "Here's the answer:"
- "I hope this helps!"

**코드/쿼리만 필요할 때**는 이런 텍스트가 파싱에 방해가 되므로 억제함.
