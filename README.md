# Prompt Validator Service

A FastAPI-based service for validating prompts against ATPL (AI Trust, Privacy, and Legal) criteria.

## Features

- **ATPL-based Validation**: Evaluates prompts against multiple compliance categories
- **Configurable LLM Integration**: Supports custom LLM proxy endpoints for semantic evaluation
- **Weighted Scoring**: Customizable category weights for tailored scoring
- **Kubernetes-Ready**: Docker container with ConfigMap/Secret support
- **RESTful API**: Clean, documented endpoints with Pydantic models

## Evaluation Categories

The service evaluates prompts across five key categories:

1. **Purpose Clarity** (default weight: 15%) - Clear definition of intent and expected outcomes
2. **Safety & Prohibited Content** (default weight: 30%) - Absence of harmful, illegal, or prohibited content
3. **Data Sensitivity & Compliance** (default weight: 25%) - Proper handling of sensitive data and regulatory compliance
4. **Trust & Provenance** (default weight: 15%) - Traceability and authenticity of information sources
5. **Agent Autonomy Bounds** (default weight: 15%) - Appropriate limits on autonomous agent actions

## Installation

### PyPI Package (Recommended)

Install the package from PyPI (once published):
```bash
pip install prompt-validator
```

Use as a library in your Python code:
```python
from prompt_validator import (
    ValidatePromptRequest,
    ValidatePromptResponse,
    ATPlSchemaLoader,
    LLMEvaluator,
    ScoringEngine,
)

# Use the components in your application
request = ValidatePromptRequest(
    prompt="Your prompt text here",
    context={"purpose": "testing"}
)
```

Or run the FastAPI service:
```bash
python -m uvicorn prompt_validator.main:app --reload
```

The service will be available at `http://localhost:8000`.

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/SentriusLLC/promptLLM.git
cd promptLLM
```

2. Install in development mode:
```bash
pip install -e .
```

3. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

4. Run the service:
```bash
python -m uvicorn prompt_validator.main:app --reload
```

The service will be available at `http://localhost:8000`.

### Docker

Build the Docker image:
```bash
docker build -t prompt-validator:latest .
```

Run the container:
```bash
docker run -p 8000:8000 \
  -e LLM_ENDPOINT="https://your-llm-proxy.example.com/v1/chat/completions" \
  -e LLM_API_KEY="your-api-key" \
  -e LLM_ENABLED="true" \
  prompt-validator:latest
```

For Keycloak or custom header authentication:
```bash
docker run -p 8000:8000 \
  -e LLM_ENDPOINT="https://your-llm-proxy.example.com/v1/chat/completions" \
  -e LLM_CUSTOM_HEADERS="X-Ztat-Token:your-keycloak-token" \
  -e LLM_ENABLED="true" \
  prompt-validator:latest
```

### Kubernetes Deployment

1. Update the ConfigMap and Secret in `k8s-deployment.yaml` with your configuration:
   - Set your LLM proxy endpoint in the ConfigMap
   - Set your API key in the Secret

2. Apply the configuration:
```bash
kubectl apply -f k8s-deployment.yaml
```

3. Verify the deployment:
```bash
kubectl get pods -l app=prompt-validator
kubectl get svc prompt-validator
```

## Configuration

All configuration is done through environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host binding |
| `PORT` | `8000` | Server port |
| `ATPL_SCHEMA_URL` | GitHub URL | URL to ATPL schema JSON |
| `LLM_ENDPOINT` | (empty) | LLM API endpoint URL |
| `LLM_API_KEY` | (empty) | API key for LLM endpoint |
| `LLM_MODEL` | `gpt-4` | Model name to use |
| `LLM_ENABLED` | `false` | Enable LLM-based evaluation |
| `LLM_CUSTOM_HEADERS` | (empty) | Custom authentication headers (see below) |
| `WEIGHT_PURPOSE` | `15` | Weight for purpose category (0-100) |
| `WEIGHT_SAFETY` | `30` | Weight for safety category (0-100) |
| `WEIGHT_COMPLIANCE` | `25` | Weight for compliance category (0-100) |
| `WEIGHT_PROVENANCE` | `15` | Weight for provenance category (0-100) |
| `WEIGHT_AUTONOMY` | `15` | Weight for autonomy category (0-100) |

**Note**: Weights should sum to 100. If they don't, they will be automatically normalized.

### Custom Authentication Headers

The `LLM_CUSTOM_HEADERS` variable supports identity management systems like Keycloak that use custom HTTP headers instead of API keys. 

**Format**: `"Header-Name:header-value,Another-Header:another-value"`

**Example for Keycloak**: 
```bash
LLM_CUSTOM_HEADERS="X-Ztat-Token:your-keycloak-token"
```

**Multiple headers**:
```bash
LLM_CUSTOM_HEADERS="X-Ztat-Token:token123,X-Custom-Auth:auth456"
```

These custom headers will be sent along with all LLM requests. If both `LLM_API_KEY` and `LLM_CUSTOM_HEADERS` are configured, both will be sent (custom headers first, then the Authorization Bearer token).

## API Endpoints

### GET /

Root endpoint - returns service information.

**Response:**
```json
{
  "service": "prompt_validator",
  "version": "0.1.0",
  "status": "running"
}
```

### GET /health

Health check endpoint for Kubernetes probes.

**Response:**
```json
{
  "status": "healthy"
}
```

### GET /criteria

List current ATPL evaluation criteria and their weights.

**Response:**
```json
{
  "criteria": [
    {
      "name": "Purpose",
      "description": "Clear definition of intent and expected outcomes",
      "weight": 15
    },
    ...
  ],
  "total_weight": 100
}
```

### POST /validate_prompt

Validate a prompt against ATPL criteria.

**Request Body:**
```json
{
  "prompt": "Generate a report on user behavior",
  "context": {
    "user_role": "analyst",
    "data_access": "anonymized"
  },
  "schema_url": "https://custom-schema-url.com/schema.json"
}
```

- `prompt` (required): The user prompt to validate
- `context` (optional): Additional context as JSON
- `schema_url` (optional): Custom ATPL schema URL

**Response:**
```json
{
  "score": 85,
  "ratings": {
    "purpose": 9,
    "safety": 8,
    "compliance": 9,
    "provenance": 8,
    "autonomy": 9
  },
  "explanation": "The prompt demonstrates clear purpose and appropriate data handling...",
  "recommendations": [
    "Consider adding explicit data retention policies",
    "Specify the intended audience for the report"
  ]
}
```

## LLM Integration

The service supports flexible integration with various LLM providers including OpenAI, custom LLM proxies, and identity management systems like Keycloak.

### Standard OpenAI-style Authentication

1. Set `LLM_ENDPOINT` to your API URL (e.g., `https://api.openai.com/v1/chat/completions`)
2. Set `LLM_API_KEY` to your API key
3. Set `LLM_ENABLED=true` to enable LLM-based evaluation
4. Optionally configure `LLM_MODEL` to specify the model name

### Keycloak / Custom Header Authentication

For LLM proxies that use managed identities with custom headers (like Keycloak with `X-Ztat-Token`):

1. Set `LLM_ENDPOINT` to your proxy URL
2. Set `LLM_CUSTOM_HEADERS` with your custom authentication headers:
   ```bash
   LLM_CUSTOM_HEADERS="X-Ztat-Token:your-keycloak-token"
   ```
3. Set `LLM_ENABLED=true` to enable LLM-based evaluation
4. Leave `LLM_API_KEY` empty if not needed

### Hybrid Authentication

You can use both custom headers and API keys simultaneously. Custom headers are sent first, followed by the `Authorization: Bearer` token if `LLM_API_KEY` is provided.

The service will send requests in OpenAI-compatible format but can work with any proxy that accepts similar requests.

If LLM evaluation is disabled or fails, the service falls back to neutral scoring with automated recommendations.

## Examples

### Example 1: Basic Validation

```bash
curl -X POST http://localhost:8000/validate_prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Summarize the customer feedback from last quarter"
  }'
```

### Example 2: Validation with Context

```bash
curl -X POST http://localhost:8000/validate_prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Analyze user purchase patterns",
    "context": {
      "data_source": "anonymized_transactions",
      "purpose": "market_research",
      "retention_period": "90_days"
    }
  }'
```

### Example 3: Get Criteria

```bash
curl http://localhost:8000/criteria
```

## Development

### Running Tests

```bash
pytest tests/
```

### Building for PyPI

To build distribution packages for PyPI:

```bash
# Install build tool
pip install build

# Build source distribution and wheel
python -m build

# Distributions will be created in dist/
# - prompt-validator-x.x.x.tar.gz (source distribution)
# - prompt_validator-x.x.x-py3-none-any.whl (wheel)
```

To publish to PyPI (maintainers only):

```bash
# Install twine
pip install twine

# Upload to PyPI
twine upload dist/*

# Or upload to TestPyPI first
twine upload --repository testpypi dist/*
```

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Architecture

```
prompt_validator/
├── __init__.py          # Package initialization
├── main.py              # FastAPI application
├── config.py            # Configuration management
├── models.py            # Pydantic request/response models
├── schema_loader.py     # ATPL schema loading
├── llm_evaluator.py     # LLM integration for semantic evaluation
└── scoring.py           # Scoring engine with weighted calculations
```

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Contributing

We encourage PRs
