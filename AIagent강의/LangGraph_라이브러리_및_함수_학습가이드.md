# LangGraph 학습 가이드 (재구성)

> Part 1. AI Agent 이해와 입문 프로젝트 순서에 따른 정리
> 기초 개념 → 실전 프로젝트 흐름으로 구성

---

# Chapter 02. LangGraph 핵심 개념

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
