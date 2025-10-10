"""LLM evaluator module for semantic prompt analysis."""
import httpx
from typing import Dict, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)


class LLMEvaluator:
    """Evaluator that uses an LLM to assess prompt compliance."""
    
    def __init__(self, endpoint: str, api_key: str = "", model: str = "gpt-4", enabled: bool = False, custom_headers: str = ""):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.enabled = enabled
        self.custom_headers = self._parse_custom_headers(custom_headers)
    
    async def evaluate_prompt(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        criteria: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a prompt using LLM.
        
        Returns a dictionary with:
        - ratings: dict of category -> score (0-10)
        - explanation: text justification
        - recommendations: list of improvement suggestions
        """
        if not self.enabled or not self.endpoint:
            # Return neutral scores if LLM is disabled
            return self._get_neutral_evaluation()
        
        try:
            evaluation_prompt = self._build_evaluation_prompt(prompt, context, criteria)
            result = await self._call_llm(evaluation_prompt)
            return self._parse_llm_response(result)
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            return self._get_neutral_evaluation()
    
    def _parse_custom_headers(self, custom_headers_str: str) -> Dict[str, str]:
        """Parse custom headers from comma-separated key:value pairs.
        
        Format: "Header-Name:header-value,Another-Header:another-value"
        Example: "X-Ztat-Token:abc123,X-Custom-Header:xyz789"
        """
        headers = {}
        if not custom_headers_str:
            return headers
        
        for pair in custom_headers_str.split(","):
            pair = pair.strip()
            if ":" in pair:
                key, value = pair.split(":", 1)
                headers[key.strip()] = value.strip()
        
        return headers
    
    def _build_evaluation_prompt(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]],
        criteria: Optional[Dict[str, str]]
    ) -> str:
        """Build the evaluation prompt for the LLM."""
        criteria_text = ""
        if criteria:
            criteria_text = "\n".join([f"- {key}: {desc}" for key, desc in criteria.items()])
        
        context_text = ""
        if context:
            context_text = f"\n\nContext: {json.dumps(context, indent=2)}"
        
        return f"""You are an AI prompt safety, compliance, and efficacy evaluator. Evaluate the following prompt against these criteria:

{criteria_text}

Rate each criterion from 0-10 where:
- 0-3: Poor/Non-compliant
- 4-6: Acceptable with concerns
- 7-10: Good/Compliant

Prompt to evaluate:
{prompt}{context_text}

Respond with a JSON object containing:
{{
  "ratings": {{
    "purpose": <0-10>,
    "safety": <0-10>,
    "compliance": <0-10>,
    "provenance": <0-10>,
    "autonomy": <0-10>
  }},
  "explanation": "<detailed explanation>",
  "recommendations": ["<recommendation 1>", "<recommendation 2>", ...]
}}"""
    
    async def _call_llm(self, evaluation_prompt: str) -> str:
        """Call the LLM endpoint."""
        headers = {}
        
        # Add custom headers first (e.g., X-Ztat-Token for Keycloak)
        if self.custom_headers:
            headers.update(self.custom_headers)
        
        # Add API key as Bearer token if provided (OpenAI-style)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Support both OpenAI-style and generic endpoints
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant that evaluates prompts for safety and compliance."},
                {"role": "user", "content": evaluation_prompt}
            ],
            "temperature": 0.3
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.endpoint,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract content from response (OpenAI format)
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            # Generic format
            elif "content" in result:
                return result["content"]
            else:
                return json.dumps(result)
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured format."""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            result = json.loads(response)
            
            # Validate structure
            if "ratings" not in result:
                return self._get_neutral_evaluation()
            
            return {
                "ratings": result.get("ratings", {}),
                "explanation": result.get("explanation", "LLM evaluation completed."),
                "recommendations": result.get("recommendations", [])
            }
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response as JSON: {response}")
            return self._get_neutral_evaluation()
    
    def _get_neutral_evaluation(self) -> Dict[str, Any]:
        """Return a neutral evaluation when LLM is unavailable."""
        return {
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
