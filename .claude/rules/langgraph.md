---
paths:
  - src/specagent/nodes/**/*.py
  - src/specagent/graph/**/*.py
---

# LangGraph Node Rules

## Node Function Signature
Every node MUST follow this exact signature:
```python
def node_name(state: GraphState) -> GraphState:
    """Docstring explaining node purpose."""
    # Implementation
    return state
```

## State Access Pattern
```python
# Read from state
question = state.get("question", "")
chunks = state.get("retrieved_chunks", [])

# Write to state (create new dict, don't mutate)
state["generation"] = answer
state["citations"] = citations
return state
```

## Structured Output with Pydantic
Define output schema as Pydantic model:
```python
class GradeResult(BaseModel):
    relevant: Literal["yes", "no"]
    confidence: float = Field(ge=0.0, le=1.0)

result = llm.with_structured_output(GradeResult).invoke(prompt)
```

## Prompt Templates
Store prompts as module constants:
```python
ROUTER_PROMPT = """You are a router for a 3GPP specification assistant.
...
"""
```

## Error Handling
Nodes should handle errors gracefully:
```python
try:
    result = llm.invoke(prompt)
except Exception as e:
    state["error"] = str(e)
    return state
```

## Dataclasses for State Objects
Use dataclasses from `graph/state.py`:
- `RetrievedChunk` - Retrieved document chunk
- `GradedChunk` - Chunk with relevance score
- `Citation` - Source citation
