# Review Code

Perform a code review of recent changes or specified files.

## Review Checklist

1. **Tests**: Do tests exist? Do they cover edge cases?
2. **Types**: Are type hints complete? Run `mypy` to verify.
3. **Patterns**: Does code follow project patterns in `.claude/rules/`?
4. **Errors**: Is error handling appropriate?
5. **Config**: Are hardcoded values moved to `settings`?
6. **Docs**: Are docstrings present and accurate?
7. **Security**: No secrets in code? API keys from environment?

## Commands to Run

```bash
# Lint
ruff check src/specagent/

# Type check
mypy src/specagent/

# Security scan
bandit -r src/specagent/

# Test coverage
pytest --cov=specagent --cov-report=term-missing
```

## Output Format

Provide feedback as:
- ✅ **Good**: Things done well
- ⚠️ **Suggestion**: Improvements to consider
- ❌ **Issue**: Problems that should be fixed
