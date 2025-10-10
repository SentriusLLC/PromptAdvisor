"""Prompt Validator - FastAPI service for ATPL-based prompt validation."""

__version__ = "0.1.0"

# Export main components for library usage
from prompt_validator.config import settings, Settings
from prompt_validator.models import (
    ValidatePromptRequest,
    ValidatePromptResponse,
    CategoryRatings,
    CriteriaResponse,
    Criterion,
)
from prompt_validator.schema_loader import ATPlSchemaLoader
from prompt_validator.llm_evaluator import LLMEvaluator
from prompt_validator.scoring import ScoringEngine

__all__ = [
    "__version__",
    "settings",
    "Settings",
    "ValidatePromptRequest",
    "ValidatePromptResponse",
    "CategoryRatings",
    "CriteriaResponse",
    "Criterion",
    "ATPlSchemaLoader",
    "LLMEvaluator",
    "ScoringEngine",
]
