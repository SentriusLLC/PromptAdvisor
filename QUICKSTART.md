# Quick Start Guide

## Local Development (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Configure Environment
```bash
cp .env.example .env
# Edit .env to add your LLM endpoint and API key if needed
```

### 3. Start the Service
```bash
python -m uvicorn prompt_validator.main:app --reload
```

The service will be available at `http://localhost:8000`

### 4. Test the Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Get Criteria
```bash
curl http://localhost:8000/criteria | python -m json.tool
```

#### Validate a Prompt
```bash
curl -X POST http://localhost:8000/validate_prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Generate a report on customer behavior",
    "context": {
      "data_source": "anonymized_transactions",
      "purpose": "market_research"
    }
  }' | python -m json.tool
```

### 5. View API Documentation
Open your browser to:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Docker Deployment

### Build and Run
```bash
# Build the image
docker build -t prompt-validator:latest .

# Run the container
docker run -p 8000:8000 \
  -e LLM_ENDPOINT="https://your-llm-proxy.example.com/v1/chat/completions" \
  -e LLM_API_KEY="your-api-key" \
  -e LLM_ENABLED="true" \
  prompt-validator:latest
```

## Kubernetes Deployment

### 1. Update Configuration
Edit `k8s-deployment.yaml` to set:
- Your LLM proxy endpoint in the ConfigMap
- Your API key in the Secret

### 2. Deploy
```bash
kubectl apply -f k8s-deployment.yaml
```

### 3. Verify
```bash
# Check pods
kubectl get pods -l app=prompt-validator

# Check service
kubectl get svc prompt-validator

# Port forward for testing
kubectl port-forward svc/prompt-validator 8000:80

# Test
curl http://localhost:8000/health
```

## Configuration Options

All settings can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_ENABLED` | `false` | Enable LLM evaluation |
| `LLM_ENDPOINT` | (empty) | Your LLM proxy URL |
| `LLM_API_KEY` | (empty) | API key for authentication |
| `LLM_MODEL` | `gpt-4` | Model name to use |
| `WEIGHT_PURPOSE` | `15` | Purpose clarity weight (0-100) |
| `WEIGHT_SAFETY` | `30` | Safety weight (0-100) |
| `WEIGHT_COMPLIANCE` | `25` | Compliance weight (0-100) |
| `WEIGHT_PROVENANCE` | `15` | Provenance weight (0-100) |
| `WEIGHT_AUTONOMY` | `15` | Autonomy weight (0-100) |

## Testing

Run the test suite:
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v
```

## Example Responses

### Criteria Response
```json
{
  "criteria": [
    {
      "name": "Purpose",
      "description": "Clear definition of intent and expected outcomes",
      "weight": 15
    },
    {
      "name": "Safety",
      "description": "Absence of harmful, illegal, or prohibited content",
      "weight": 30
    },
    {
      "name": "Compliance",
      "description": "Proper handling of sensitive data and regulatory compliance",
      "weight": 25
    },
    {
      "name": "Provenance",
      "description": "Traceability and authenticity of information sources",
      "weight": 15
    },
    {
      "name": "Autonomy",
      "description": "Appropriate limits on autonomous agent actions",
      "weight": 15
    }
  ],
  "total_weight": 100
}
```

### Validation Response
```json
{
  "score": 70,
  "ratings": {
    "purpose": 7,
    "safety": 7,
    "compliance": 7,
    "provenance": 7,
    "autonomy": 7
  },
  "explanation": "Automated evaluation completed. LLM evaluation not available or disabled.",
  "recommendations": [
    "Consider reviewing prompt for clarity and specificity",
    "Ensure no sensitive data is exposed in the prompt",
    "Verify that the prompt aligns with intended use cases"
  ]
}
```

## Troubleshooting

### Service won't start
- Check that port 8000 is not already in use
- Verify all dependencies are installed: `pip install -r requirements.txt`

### LLM evaluation not working
- Ensure `LLM_ENABLED=true`
- Verify `LLM_ENDPOINT` is set and accessible
- Check that `LLM_API_KEY` is correct (if required)
- The service will fall back to neutral scoring if LLM is unavailable

### ATPL schema fails to load
- The service uses a default schema as fallback
- Check network connectivity to GitHub
- Verify the URL in `ATPL_SCHEMA_URL` is accessible

### Docker build fails with SSL errors
- See `DOCKER_NOTES.md` for solutions
- This is typically a CI/CD infrastructure issue
- The Dockerfile is correctly structured for standard environments

## Next Steps

1. **Configure your LLM proxy**: Set the environment variables for your specific LLM endpoint
2. **Customize scoring weights**: Adjust category weights based on your compliance priorities
3. **Integrate with your application**: Use the `/validate_prompt` endpoint in your AI pipeline
4. **Monitor usage**: The service logs all evaluations for audit purposes
5. **Scale as needed**: Use Kubernetes replicas to handle higher traffic

For more details, see the full [README.md](README.md).
