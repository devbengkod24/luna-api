# Tests

Directory untuk unit tests dan integration tests.

## Structure

```
tests/
├── unit/           # Unit tests untuk individual components
├── integration/    # Integration tests untuk API endpoints
└── fixtures/       # Test fixtures dan mock data
```

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

