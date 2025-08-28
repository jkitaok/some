from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from some.progress import as_completed_with_tqdm
from typing import Any, Dict, List, Optional, Tuple, Callable
import logging
import os
import time

from openai import OpenAI

from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()


# Environment variables are loaded above; defaults are resolved inside provider classes.

# Pluggable registry for custom providers
LANGUAGE_MODEL_REGISTRY: Dict[str, Callable[..., "BaseLanguageModel"]] = {}

def register_language_model(name: str, factory: Callable[..., "BaseLanguageModel"]) -> None:
    """Register a language model factory under a provider name.

    Example:
        from some.inference import register_language_model, BaseLanguageModel

        class MyLM(BaseLanguageModel):
            ...

        register_language_model("myprovider", lambda **kw: MyLM(**kw))
    """
    LANGUAGE_MODEL_REGISTRY[name.lower()] = factory

# Default model registry per provider (overridable at runtime)
DEFAULT_MODEL_REGISTRY: Dict[str, str] = {
    "openai": "gpt-5-nano",
    "ollama": "qwen3:4b-instruct",
}

def set_default_model(provider: str, model: str) -> None:
    """Set/override the default model for a provider (e.g., "openai", "ollama")."""
    DEFAULT_MODEL_REGISTRY[provider.lower()] = model

def get_default_model(provider: str) -> Optional[str]:
    """Get the default model for a provider (or None if not registered)."""
    return DEFAULT_MODEL_REGISTRY.get(provider.lower())



class BaseLanguageModel(ABC):
    """Abstract interface for language models used by the extraction system.

    Implementors fill in `generate` using the provider's API. All models expose
    a `model_id` attribute set during __init__, never in `generate`.
    """

    def __init__(self, *, model: Optional[str] = None) -> None:
        # Subclasses may override defaults; this stores the requested model.
        self.model_id: Optional[str] = model

    @abstractmethod
    def build_messages(self, prompt_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert generic prompt data to model-specific message format.

        Args:
            prompt_data: Dict containing:
                - prompt_text: str - The formatted text prompt
                - image_path: Optional[str] - Local path to image file if needed
                - response_format: Optional[BaseModel] - Pydantic model for structured output
                - result_key: str - Key name for storing results

        Returns:
            List of message dicts in the format expected by this language model's API
        """

    @abstractmethod
    def generate(
        self,
        inputs: List[Dict[str, Any]],
        *,
        max_workers: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], int, float]:
        """Run batch generation over inputs.

        Returns (results, effective_max_workers, total_inference_time)
        where each result is typically a dict like:
        {
          "input_tokens": int,
          "output_tokens": int,
          <result_key>: Any | None,  # as provided by prompt builder
          "error": Optional[str]
        }
        """


class OpenAILanguageModel(BaseLanguageModel):
    def __init__(self, *, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        super().__init__(model=model)
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for provider 'openai'.")
        # Resolve base URL default here
        resolved_base = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.client = OpenAI(base_url=resolved_base, api_key=key)
        # Default model if not provided at init
        if self.model_id is None:
            self.model_id = os.getenv("OPENAI_MODEL", get_default_model("openai") or "gpt-5-nano")

    def build_messages(self, prompt_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert generic prompt data to OpenAI message format."""
        from .media import encode_base64_content_from_path, get_image_mime_type

        prompt_text = prompt_data.get("prompt_text", "")
        image_path = prompt_data.get("image_path")

        if image_path:
            # Handle multimodal content (text + image)
            base64_image = encode_base64_content_from_path(image_path)
            mime_type = get_image_mime_type(image_path)

            return [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        else:
            # Text-only content
            return [
                {
                    "role": "user",
                    "content": prompt_text
                }
            ]

    def _generate_single(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Build messages using the new format
        messages = self.build_messages(payload)
        response_format = payload.get("response_format")  # prompt builders should supply when structured output is desired
        result_key = payload.get("result_key", "result")

        completion = self.client.beta.chat.completions.parse(
            model=self.model_id,
            messages=messages,
            response_format=response_format,
        )

        return {
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
            result_key: completion.choices[0].message.parsed.model_dump(),
        }

    def generate(self, inputs: List[Dict[str, Any]], *, max_workers: Optional[int] = None):
        if not inputs:
            return [], 0, 0.0

        # Reasonable cap to avoid high parallelism on APIs
        if max_workers is None:
            cpu_count = os.cpu_count()
            max_workers = max(1, (cpu_count or 4) - 1)
        max_workers = min(max_workers, len(inputs), 10)

        results: List[Dict[str, Any]] = [{}] * len(inputs)

        def task(idx: int, item: Dict[str, Any]):
            try:
                res = self._generate_single(item)
                results[idx] = res
            except Exception as e:
                logging.error("OpenAI inference error for item %d: %s", idx, e)
                # Use result_key if present to keep downstream consistent
                result_key = item.get("result_key", "result")
                results[idx] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    result_key: None,
                    "error": str(e),
                }

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(task, i, item) for i, item in enumerate(inputs)]
            for _ in as_completed_with_tqdm(
                futures,
                total=len(futures),
                desc="LLM",
                unit="item",
                colour="magenta",
            ):
                pass

        total_inference_time = time.time() - start_time

        return results, max_workers, total_inference_time


class OllamaLanguageModel(BaseLanguageModel):
    def __init__(self, *, base_url: Optional[str] = None, model: Optional[str] = None):
        super().__init__(model=model)
        # Ollama-compatible OpenAI client: api_key can be any string
        resolved_base = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.client = OpenAI(base_url=resolved_base, api_key="ollama")
        if self.model_id is None:
            self.model_id = os.getenv("OLLAMA_MODEL", get_default_model("ollama") or "qwen3:4b-instruct")

    def build_messages(self, prompt_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert generic prompt data to Ollama message format (OpenAI-compatible)."""
        from .media import encode_base64_content_from_path, get_image_mime_type

        prompt_text = prompt_data.get("prompt_text", "")
        image_path = prompt_data.get("image_path")

        if image_path:
            # Handle multimodal content (text + image) - same format as OpenAI
            base64_image = encode_base64_content_from_path(image_path)
            mime_type = get_image_mime_type(image_path)

            return [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        else:
            # Text-only content
            return [
                {
                    "role": "user",
                    "content": prompt_text
                }
            ]

    def _generate_single(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Build messages using the new format
        messages = self.build_messages(payload)
        response_format = payload.get("response_format")  # prompt builders should supply when structured output is desired
        result_key = payload.get("result_key", "result")

        completion = self.client.beta.chat.completions.parse(
            model=self.model_id,
            messages=messages,
            response_format=response_format,
        )

        return {
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
            result_key: completion.choices[0].message.parsed.model_dump(),
        }

    def generate(self, inputs: List[Dict[str, Any]], *, max_workers: Optional[int] = None):
        if not inputs:
            return [], 0, 0.0

        if max_workers is None:
            cpu_count = os.cpu_count()
            max_workers = max(1, (cpu_count or 4) - 1)
        # keep lower cap by default for local setups
        max_workers = min(max_workers, len(inputs), 6)

        results: List[Dict[str, Any]] = [{}] * len(inputs)

        def task(idx: int, item: Dict[str, Any]):
            try:
                res = self._generate_single(item)
                results[idx] = res
            except Exception as e:
                logging.error("Ollama inference error for item %d: %s", idx, e)
                result_key = item.get("result_key", "result")
                results[idx] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    result_key: None,
                    "error": str(e),
                }

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(task, i, item) for i, item in enumerate(inputs)]
            for _ in as_completed_with_tqdm(
                futures,
                total=len(futures),
                desc="LLM",
                unit="item",
                colour="magenta",
            ):
                pass

        total_inference_time = time.time() - start_time

        return results, max_workers, total_inference_time


def get_language_model(*, provider: str, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None) -> BaseLanguageModel:
    provider = (provider or "openai").lower()

    # Custom registered providers take precedence
    if provider in LANGUAGE_MODEL_REGISTRY:
        return LANGUAGE_MODEL_REGISTRY[provider](api_key=api_key, base_url=base_url, model=model)

    if provider == "openai":
        return OpenAILanguageModel(api_key=api_key, base_url=base_url, model=model)
    if provider == "ollama":
        return OllamaLanguageModel(base_url=base_url, model=model)
    raise ValueError(f"Unsupported provider: {provider}")

