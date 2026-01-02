---
paths:
  - src/specagent/api/**/*.py
---

# API Development Rules

## Endpoint Pattern
```python
@router.post("/endpoint", response_model=ResponseModel)
async def endpoint_name(request: RequestModel) -> ResponseModel:
    """OpenAPI docstring."""
    # Implementation
```

## Request/Response Models
Define in `api/models.py` with full validation:
```python
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    verbose: bool = Field(default=False)
```

## Error Handling
Use HTTPException with structured detail:
```python
raise HTTPException(
    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    detail={"error": "off_topic", "message": "..."}
)
```

## Health Check
The `/health` endpoint is required for k8s probes.
