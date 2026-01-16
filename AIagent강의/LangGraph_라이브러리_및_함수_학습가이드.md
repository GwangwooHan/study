# LangGraph 학습 가이드 (재구성)

> Part 1. AI Agent 이해와 입문 프로젝트 순서에 따른 정리
> 기초 개념 → 실전 프로젝트 흐름으로 구성

---

# Chapter 02. LangGraph 핵심 개념

## 01. LangChain vs LangGraph 구별법

### 두 라이브러리의 관계

| 라이브러리 | 역할 | 비유 |
|-----------|------|------|
| **LangChain** | LLM 호출, 프롬프트, 체인 구성 | 개별 작업 도구 |
| **LangGraph** | 복잡한 워크플로우, 상태 관리, 분기/반복 | 작업 흐름 설계도 |

> LangGraph는 LangChain 위에서 동작합니다. LangChain의 LLM, 프롬프트 등을 LangGraph의 노드 안에서 사용합니다.

### LangChain 특징 및 장/단점

**특징:**
- LLM과의 상호작용을 단순화하는 **추상화 레이어**
- 프롬프트 템플릿, 출력 파서, 메모리 등 **모듈화된 컴포넌트** 제공
- `|` (파이프) 연산자로 컴포넌트를 **선형적으로 연결**
- 다양한 LLM 프로바이더 지원 (OpenAI, Anthropic, HuggingFace 등)

**장점:**
| 장점 | 설명 |
|------|------|
| **빠른 프로토타이핑** | 몇 줄의 코드로 LLM 앱 구현 가능 |
| **풍부한 통합** | 벡터DB, 문서 로더, 임베딩 등 다양한 도구 지원 |
| **직관적인 문법** | `prompt | llm | parser` 형태로 읽기 쉬움 |
| **커뮤니티 생태계** | 방대한 문서, 예제, 서드파티 통합 |

**단점:**
| 단점 | 설명 |
|------|------|
| **선형 흐름 한계** | 복잡한 분기/반복 로직 구현이 어려움 |
| **상태 관리 부재** | 체인 간 상태 공유/추적이 제한적 |
| **디버깅 어려움** | 체인이 길어지면 중간 상태 파악이 힘듦 |
| **과도한 추상화** | 단순한 작업에도 불필요한 복잡성 추가 가능 |

### LangGraph 특징 및 장/단점

**특징:**
- **그래프 기반** 워크플로우 설계 (노드 + 엣지)
- **명시적 상태 관리** (TypedDict + Reducer)
- **조건부 분기와 반복** 지원
- **체크포인트/메모리** 내장으로 대화 지속성 제공
- Human-in-the-loop 패턴 지원

**장점:**
| 장점 | 설명 |
|------|------|
| **복잡한 흐름 제어** | 분기, 반복, 병렬 실행 등 자유로운 워크플로우 |
| **상태 추적 용이** | State가 명시적이라 디버깅/모니터링 편리 |
| **재사용성** | 노드를 독립적으로 테스트하고 조합 가능 |
| **장기 실행 지원** | 체크포인트로 중단/재개 가능 |
| **시각화** | 그래프 구조를 Mermaid 다이어그램으로 확인 |

**단점:**
| 단점 | 설명 |
|------|------|
| **학습 곡선** | State, Reducer, Edge 개념 이해 필요 |
| **보일러플레이트** | 단순 작업에도 State 정의, 노드 등록 필요 |
| **설정 복잡도** | 간단한 체인보다 초기 설정 코드가 많음 |
| **오버엔지니어링 위험** | 단순한 문제에 과도한 구조 적용 가능성 |

### 언제 무엇을 사용할까?

| 상황 | 권장 | 이유 |
|------|------|------|
| 단순 질의응답 | LangChain | 파이프 체인으로 충분 |
| RAG 기본 구현 | LangChain | 문서 로더 + 벡터DB 통합 용이 |
| 조건부 분기 필요 | **LangGraph** | 조건부 엣지로 명확한 흐름 제어 |
| 반복 로직 필요 | **LangGraph** | 루프백 엣지로 반복 구현 |
| Multi-Agent 시스템 | **LangGraph** | 에이전트 간 상태 공유/조정 |
| 장기 실행 작업 | **LangGraph** | 체크포인트로 중단/재개 |
| Human-in-the-loop | **LangGraph** | 사용자 승인 노드 삽입 용이 |
| 빠른 프로토타입 | LangChain | 최소한의 코드로 검증 |

**핵심 원칙:**
> 단순하게 시작하고, 복잡해지면 전환하라
> - **LangChain**으로 시작: 빠른 검증
> - **LangGraph**로 전환: 분기/반복/상태 관리가 필요해질 때

### 빠른 구별법: import 문 확인

```python
# LangChain만 사용
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# LangGraph 사용 시 추가됨
from langgraph.graph import StateGraph  # ← 이게 있으면 LangGraph
```

### 코드 패턴 비교

| 구분 | LangChain | LangGraph |
|------|-----------|-----------|
| **import** | `from langchain_*` | `from langgraph.*` |
| **핵심 클래스** | `ChatPromptTemplate`, `LLMChain` | `StateGraph`, `MessageGraph` |
| **연결 방식** | `|` (파이프) | `.add_node()`, `.add_edge()` |
| **State 관리** | 없거나 단순 dict | `TypedDict` + `Annotated` |
| **실행** | `chain.invoke()` | `graph.compile().invoke()` |

### 코드 예시 비교

**LangChain 패턴:**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([...])
llm = ChatOpenAI(model="gpt-4o")

chain = prompt | llm  # 파이프로 연결
result = chain.invoke({"question": "..."})
```

**LangGraph 패턴:**
```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    question: str
    answer: str

graph = StateGraph(State)
graph.add_node("process", process_function)
graph.add_edge(START, "process")
graph.add_edge("process", END)

app = graph.compile()
result = app.invoke({"question": "..."})
```

### 키워드로 빠르게 파악하기

| 키워드 | 의미 |
|--------|------|
| `StateGraph` | LangGraph |
| `add_node`, `add_edge` | LangGraph |
| `compile()` | LangGraph |
| `TypedDict` + `Annotated[..., add]` | LangGraph State 패턴 |
| `|` (파이프 연산자) | LangChain 체인 |
| `ChatPromptTemplate` | LangChain (LangGraph에서도 사용) |

### 혼합 사용 패턴

실제 프로젝트에서는 **둘을 함께 사용**합니다:

```python
# LangChain: 프롬프트와 LLM 정의
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([...])
llm = ChatOpenAI(model="gpt-4o")
chain = prompt | llm  # LangChain 체인

# LangGraph: 워크플로우 정의
from langgraph.graph import StateGraph

def my_node(state):
    # 노드 안에서 LangChain 체인 사용
    result = chain.invoke({"question": state["question"]})
    return {"answer": result.content}

graph = StateGraph(State)
graph.add_node("process", my_node)  # LangChain 체인을 LangGraph 노드에서 실행
```

---

## 02. LangGraph 필수 구성요소

LangGraph는 **3가지 핵심 요소**로 구성됩니다:

### 1. State (상태)

그래프 전체에서 공유되는 **데이터 저장소**입니다.

```python
from typing import TypedDict

class State(TypedDict):
    messages: list      # 대화 내용 저장
    user_name: str      # 사용자 이름
    counter: int        # 숫자 카운터
```

**핵심 포인트**: State는 그래프의 모든 노드가 읽고 쓸 수 있는 공유 데이터입니다.

### 2. Node (노드)

State를 받아서 처리하고, 업데이트할 내용을 반환하는 **함수**입니다.

```python
def my_node(state: State) -> dict:
    # 1. state에서 데이터 읽기
    current_count = state["counter"]

    # 2. 처리 로직
    new_count = current_count + 1

    # 3. 업데이트할 부분만 반환
    return {"counter": new_count}
```

**핵심 포인트**: 노드는 전체 State를 받지만, 변경할 필드만 반환합니다.

### 3. Edge (엣지)

노드와 노드를 연결하는 **화살표**입니다.

```python
graph_builder.add_edge("node_a", "node_b")  # a 다음에 b 실행
```

---

## 03. 그래프의 상태 업데이트

### Reducer(리듀서)란?

State 필드가 업데이트될 때 **어떻게 업데이트할지** 정하는 규칙입니다.

| 상황 | 동작 |
|------|------|
| Reducer 없음 | 새 값으로 **덮어쓰기** |
| Reducer 있음 | 기존 값과 새 값을 **합치기** |

### Reducer 적용 방법: Annotated

```python
from typing import Annotated
from operator import add

class State(TypedDict):
    # Reducer 없음 - 덮어쓰기
    name: str

    # Reducer 있음 - 리스트 합치기
    items: Annotated[list, add]
```

### 예제: Reducer 동작 비교

```python
# 초기 상태
state = {"name": "철수", "items": ["사과"]}

# 노드가 반환한 값
return {"name": "영희", "items": ["바나나"]}

# 결과
# name: "영희"              (덮어쓰기)
# items: ["사과", "바나나"]  (합치기)
```

### add_messages 리듀서

대화 메시지를 누적하기 위한 특수 리듀서입니다.

```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

**왜 필요한가?**: 챗봇에서 이전 대화 내용을 유지하기 위함입니다.

---

## 04. 노드와 엣지 연결

### StateGraph 생성

```python
from langgraph.graph import StateGraph, START, END

# 1. State 정의
class State(TypedDict):
    value: str

# 2. 그래프 빌더 생성
graph_builder = StateGraph(State)
```

### 노드 추가

```python
def node_a(state: State):
    return {"value": "A 처리됨"}

def node_b(state: State):
    return {"value": state["value"] + " -> B 처리됨"}

graph_builder.add_node("a", node_a)
graph_builder.add_node("b", node_b)
```

### 엣지 연결

```python
# 순차 연결
graph_builder.add_edge(START, "a")  # 시작 -> a
graph_builder.add_edge("a", "b")    # a -> b
graph_builder.add_edge("b", END)    # b -> 종료
```

### 병렬 실행

하나의 노드에서 여러 노드로 동시에 연결하면 병렬 실행됩니다.

```python
graph_builder.add_edge(START, "a")
graph_builder.add_edge(START, "b")  # a와 b가 동시에 실행
graph_builder.add_edge(["a", "b"], "c")  # 둘 다 끝나면 c 실행
```

### 그래프 컴파일 및 실행

```python
# 컴파일 (실행 가능한 그래프로 변환)
graph = graph_builder.compile()

# 실행
result = graph.invoke({"value": "시작"})
print(result["value"])  # 최종 결과
```

---

## 05. 조건과 반복

### 조건부 엣지 (Conditional Edge)

상황에 따라 다른 노드로 이동합니다.

```python
def route_function(state: State) -> str:
    """다음 노드 이름을 반환"""
    if state["counter"] < 5:
        return "process"  # process 노드로
    else:
        return "finish"   # finish 노드로

graph_builder.add_conditional_edges(
    "check",           # 어디서 분기할지
    route_function,    # 분기 결정 함수
    {                  # 반환값 -> 노드 매핑
        "process": "process",
        "finish": "finish"
    }
)
```

### 반복 (Loop)

엣지를 이전 노드로 연결하면 반복됩니다.

```python
def route(state: State):
    if len(state["items"]) < 5:
        return "add_item"  # 다시 아이템 추가
    else:
        return END         # 종료

graph_builder.add_conditional_edges("check", route)
graph_builder.add_edge("add_item", "check")  # 루프백
```

### 재귀 제한

무한 루프 방지를 위해 제한을 설정할 수 있습니다.

```python
graph = graph_builder.compile()
result = graph.invoke(
    {"items": []},
    {"recursion_limit": 10}  # 최대 10번 반복
)
```

---

# Chapter 03. LangGraph 입문 프로젝트

## 01. Basic - 가장 기본적인 챗봇 만들기

### 필요한 라이브러리

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
```

### MessagesState란?

메시지 리스트를 자동 관리하는 미리 정의된 State입니다.

```python
# MessagesState는 내부적으로 이렇게 정의됨
class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]

# 사용법: 그대로 사용하거나 확장
class State(MessagesState):
    extra_field: str  # 필요시 필드 추가
```

### LLM 설정

```python
llm = ChatOpenAI(model="gpt-4o")
```

### 챗봇 노드 작성

```python
def chatbot(state: State):
    # state["messages"]에 전체 대화 내역이 있음
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

### MemorySaver - 대화 기억하기

```python
# 메모리 저장소 생성
memory = MemorySaver()

# 그래프에 연결
graph = graph_builder.compile(checkpointer=memory)

# 대화 세션 구분용 설정
config = {"configurable": {"thread_id": "user_123"}}

# 같은 thread_id로 호출하면 이전 대화 기억
graph.invoke({"messages": [{"role": "user", "content": "내 이름은 철수야"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "내 이름이 뭐야?"}]}, config)
# -> "철수"를 기억함!
```

### 전체 코드

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

# 1. LLM 설정
llm = ChatOpenAI(model="gpt-4o")

# 2. 노드 함수
def chatbot(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

# 3. 그래프 구성
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 4. 메모리와 함께 컴파일
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# 5. 실행
config = {"configurable": {"thread_id": "1"}}
response = graph.invoke(
    {"messages": [{"role": "user", "content": "안녕!"}]},
    config
)
```

---

## 02. Tool Calling - 웹검색을 하는 챗봇 만들기

### Tool이란?

LLM이 필요할 때 호출할 수 있는 외부 기능입니다.

### Tool 정의하기

```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search the web for information."""  # 이 설명을 LLM이 읽음
    # 검색 로직...
    return "검색 결과"
```

**중요**: docstring(설명)이 있어야 LLM이 언제 이 Tool을 쓸지 판단합니다.

### LLM에 Tool 연결하기

```python
tools = [search]
llm_with_tools = llm.bind_tools(tools)
```

**주의**: `bind_tools()`는 Tool을 실행하지 않습니다. LLM이 "이 Tool을 이 인자로 호출해"라고 알려주기만 합니다.

### ToolNode - Tool 실행하기

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)
```

ToolNode가 하는 일:
1. AI 메시지에서 `tool_calls` 정보 추출
2. 해당 Tool 함수 실행
3. 결과를 `ToolMessage`로 반환

### Tool Calling 라우팅

```python
def route_tools(state: State):
    ai_message = state["messages"][-1]

    # tool_calls가 있으면 Tool 실행 노드로
    if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
        return "tools"
    return END  # 없으면 종료
```

### 전체 그래프 구조

```
START -> chatbot <-> tools
            |
            v
           END
```

---

## 03. State - 할 일 목록을 관리하는 챗봇 만들기

### 복잡한 State 정의

```python
from typing import Annotated
from operator import add

class TodoState(TypedDict):
    user_input: str                           # 덮어쓰기
    flag: str                                 # 덮어쓰기
    todo_list: list[str]                      # 덮어쓰기
    completed_list: Annotated[list[str], add] # 누적
```

### Reducer 동작 비교

```python
# 노드 반환값
return {
    "todo_list": ["새 할일"],           # 기존 목록 대체
    "completed_list": ["완료된 항목"]    # 기존 목록에 추가
}
```

---

## 04. Structured Output - 원하는 형태로 논문 정보 반환하기

### Pydantic이란?

데이터 형식을 정의하고 검증하는 라이브러리입니다.

### 출력 형식 정의

```python
from pydantic import BaseModel, Field

class PaperInfo(BaseModel):
    title: str = Field(description="논문 제목")
    authors: list[str] = Field(description="저자 목록")
    year: int = Field(description="출판 연도")
```

### with_structured_output 사용

```python
# 구조화된 출력 모델 생성
structured_llm = llm.with_structured_output(PaperInfo)

# 호출하면 PaperInfo 객체 반환
result = structured_llm.invoke("BERT 논문에 대해 알려줘")

print(type(result))    # <class 'PaperInfo'>
print(result.title)    # "BERT: Pre-training of Deep..."
```

---

## 05. 문법 교정과 번역기능이 있는 영어 회화 챗봇 만들기

### 확장된 State 정의

```python
from langgraph.graph import MessagesState

class State(MessagesState):
    is_correct: bool           # 문장 정확성
    corrected_sentence: str    # 교정된 문장
    feedback: str              # 피드백 내용
```

### ChatPromptTemplate 사용

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a translator."),
    ("user", "Translate: {text}")
])

# 변수 채우기
messages = prompt.format_messages(text="안녕하세요")
response = llm.invoke(messages)
```

### 언어 감지 라우팅

```python
import re

def route_function(state: State):
    last_message = state["messages"][-1].content

    # 한글 포함 여부로 분기
    if re.search(r"[가-힣]", last_message):
        return "translation"
    else:
        return "correction"
```

---

# 핵심 개념 요약표

| 개념 | 설명 | 예시 |
|------|------|------|
| State | 공유 데이터 저장소 | `class State(TypedDict)` |
| Node | State 처리 함수 | `def chatbot(state: State)` |
| Edge | 노드 연결 | `add_edge("a", "b")` |
| Reducer | 업데이트 방식 정의 | `Annotated[list, add]` |
| add_messages | 메시지 누적 리듀서 | 대화 기록 유지 |
| MessagesState | 메시지용 기본 State | 챗봇 기본 템플릿 |
| MemorySaver | 대화 기억 | Multi-turn 대화 |
| bind_tools | Tool 연결 | `llm.bind_tools([...])` |
| ToolNode | Tool 실행 노드 | Tool 자동 실행 |
| with_structured_output | 형식 강제 | Pydantic 객체 반환 |

---

# 자주 쓰는 코드 패턴

### 기본 대화 루프

```python
config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("User: ")
    if user_input in ["q", "quit"]:
        break

    response = graph.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config
    )
    print("AI:", response["messages"][-1].content)
```

### 그래프 시각화

```python
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```
