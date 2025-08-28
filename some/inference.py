from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from some.progress import as_completed_with_tqdm
from typing import Any, Dict, List, Optional, Tuple, Callable, TYPE_CHECKING

# Type checking import to avoid circular imports
if TYPE_CHECKING:
    from some.metrics import LLMMetricsCollector
import logging
import os
import time

from openai import OpenAI

from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# Try to import instructor, but make it optional
try:
    import instructor
    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False


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
    "instructor": "openai/gpt-4o-mini",
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

    def __init__(self, *, model: Optional[str] = None, supported_modalities: Optional[List[str]] = None) -> None:
        # Subclasses may override defaults; this stores the requested model.
        self.model_id: Optional[str] = model
        self.supported_modalities = supported_modalities or ["text"]

    def supports_modality(self, modality: str) -> bool:
        """Check if the model supports a specific modality."""
        return modality in self.supported_modalities

    def get_available_modalities(self, prompt_data: Dict[str, Any]) -> List[str]:
        """Determine which modalities are present and supported for this input."""
        available = ["text"]  # Text is always present

        if prompt_data.get("image_path") or prompt_data.get("image_url"):
            if self.supports_modality("vision"):
                available.append("vision")

        if prompt_data.get("audio_path") or prompt_data.get("audio_url"):
            if self.supports_modality("audio"):
                available.append("audio")

        return available

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
        metrics_collector: Optional["LLMMetricsCollector"] = None,
    ) -> Tuple[List[Dict[str, Any]], int, float]:
        """Run batch generation over inputs.

        Args:
            inputs: List of input dictionaries for generation
            max_workers: Maximum number of workers for parallel processing
            metrics_collector: Optional LLMMetricsCollector to automatically track inference time

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
        # Determine supported modalities based on model
        modalities = ["text"]
        if model and ("gpt-4" in model.lower() or "gpt-5" in model.lower()):
            modalities.append("vision")
        if model and "audio" in model.lower():
            modalities.extend(["audio", "vision"])  # Audio models typically support vision too

        super().__init__(model=model, supported_modalities=modalities)
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
        """Convert generic prompt data to OpenAI message format with multi-modal support."""
        from .media import encode_base64_content_from_path, get_image_mime_type

        prompt_text = prompt_data.get("prompt_text", "")
        image_path = prompt_data.get("image_path")
        image_url = prompt_data.get("image_url")
        audio_path = prompt_data.get("audio_path")
        audio_url = prompt_data.get("audio_url")

        # Get available modalities for this input
        available_modalities = self.get_available_modalities(prompt_data)

        # Build content array starting with text
        content = [{"type": "text", "text": prompt_text}]

        # Add vision content if available and supported
        if "vision" in available_modalities:
            if image_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            elif image_path:
                base64_image = encode_base64_content_from_path(image_path)
                mime_type = get_image_mime_type(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                })

        # Add audio content if available and supported
        if "audio" in available_modalities:
            if audio_url:
                # For audio URLs, we include reference in text for now
                # OpenAI's audio API handles the actual audio processing
                content[0]["text"] += f"\n\n[Audio URL: {audio_url}]"
            elif audio_path:
                content[0]["text"] += f"\n\n[Audio file: {audio_path}]"

        # Return single message with multi-modal content
        if len(content) == 1:
            # Text-only, use simple format
            return [{"role": "user", "content": prompt_text}]
        else:
            # Multi-modal content
            return [{"role": "user", "content": content}]

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

    def generate(self, inputs: List[Dict[str, Any]], *, max_workers: Optional[int] = None, metrics_collector: Optional["LLMMetricsCollector"] = None):
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

        # Automatically add inference time to metrics collector if provided
        if metrics_collector is not None:
            metrics_collector.add_inference_time(total_inference_time)

        return results, max_workers, total_inference_time


class OllamaLanguageModel(BaseLanguageModel):
    def __init__(self, *, base_url: Optional[str] = None, model: Optional[str] = None):
        # Ollama typically supports vision for many models
        modalities = ["text", "vision"]
        super().__init__(model=model, supported_modalities=modalities)
        # Ollama-compatible OpenAI client: api_key can be any string
        resolved_base = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.client = OpenAI(base_url=resolved_base, api_key="ollama")
        if self.model_id is None:
            self.model_id = os.getenv("OLLAMA_MODEL", get_default_model("ollama") or "qwen3:4b-instruct")

    def build_messages(self, prompt_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert generic prompt data to Ollama message format (OpenAI-compatible) with multi-modal support."""
        from .media import encode_base64_content_from_path, get_image_mime_type

        prompt_text = prompt_data.get("prompt_text", "")
        image_path = prompt_data.get("image_path")
        image_url = prompt_data.get("image_url")

        # Get available modalities for this input
        available_modalities = self.get_available_modalities(prompt_data)

        # Build content array starting with text
        content = [{"type": "text", "text": prompt_text}]

        # Add vision content if available and supported
        if "vision" in available_modalities:
            if image_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            elif image_path:
                base64_image = encode_base64_content_from_path(image_path)
                mime_type = get_image_mime_type(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                })

        # Return single message with multi-modal content
        if len(content) == 1:
            # Text-only, use simple format
            return [{"role": "user", "content": prompt_text}]
        else:
            # Multi-modal content
            return [{"role": "user", "content": content}]

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

    def generate(self, inputs: List[Dict[str, Any]], *, max_workers: Optional[int] = None, metrics_collector: Optional["LLMMetricsCollector"] = None):
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

        # Automatically add inference time to metrics collector if provided
        if metrics_collector is not None:
            metrics_collector.add_inference_time(total_inference_time)

        return results, max_workers, total_inference_time


class InstructorLanguageModel(BaseLanguageModel):
    """Language model using the instructor library for structured output.

    Supports multiple providers through instructor.from_provider().
    Provides structured output generation with Pydantic models.

    Example usage:
        # Initialize with specific provider and model
        lm = InstructorLanguageModel(provider="anthropic", model="claude-3-haiku-20240307")

        # Or use with get_language_model
        lm = get_language_model(provider="instructor", model="openai/gpt-4o-mini")

        # Generate structured output
        results, workers, timing = lm.generate(inputs)
    """

    def __init__(self, *, model: Optional[str] = None, provider: Optional[str] = None, **kwargs) -> None:
        """Initialize the InstructorLanguageModel.

        Args:
            model: Model name, can include provider prefix (e.g., "openai/gpt-4o-mini")
            provider: Provider name (e.g., "openai", "anthropic", "ollama")
            **kwargs: Additional arguments passed to instructor.from_provider()
        """
        if not INSTRUCTOR_AVAILABLE:
            raise RuntimeError("instructor library is required for InstructorLanguageModel. Install with: pip install instructor")

        # Default to OpenAI GPT-4o-mini if no model specified
        default_model = "openai/gpt-4o-mini"
        if provider:
            default_model = f"{provider}/{model or 'gpt-4o-mini'}"
        elif model and '/' not in model:
            # If model doesn't contain provider, assume openai
            default_model = f"openai/{model}"
        elif model:
            default_model = model

        # Determine supported modalities based on model
        modalities = ["text", "vision"]  # Instructor typically supports vision
        if "audio" in default_model.lower():
            modalities.append("audio")

        super().__init__(model=default_model, supported_modalities=modalities)

        # Initialize instructor client
        try:
            self.client = instructor.from_provider(self.model_id, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize instructor client for model '{self.model_id}': {e}")

    def build_messages(self, prompt_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert generic prompt data to instructor-compatible message format with multi-modal support.

        Args:
            prompt_data: Dict containing:
                - prompt_text: str - The formatted text prompt
                - image_path: Optional[str] - Local path to image file if needed
                - image_url: Optional[str] - URL to image file
                - audio_path: Optional[str] - Local path to audio file
                - audio_url: Optional[str] - URL to audio file
                - response_format: Optional[BaseModel] - Pydantic model for structured output
                - result_key: str - Key name for storing results

        Returns:
            List of message dicts in instructor format
        """
        prompt_text = prompt_data.get("prompt_text", "")
        image_path = prompt_data.get("image_path")
        image_url = prompt_data.get("image_url")
        audio_path = prompt_data.get("audio_path")
        audio_url = prompt_data.get("audio_url")

        # Get available modalities for this input
        available_modalities = self.get_available_modalities(prompt_data)

        # Build content array starting with text
        content = [prompt_text]

        # Add vision content if available and supported
        if "vision" in available_modalities:
            try:
                if INSTRUCTOR_AVAILABLE:
                    if image_url:
                        from instructor.multimodal import Image
                        content.append(Image.from_url(image_url))
                    elif image_path:
                        from instructor.multimodal import Image
                        content.append(Image.from_path(image_path))
            except Exception as e:
                logging.warning(f"Failed to load image: {e}")

        # Add audio content if available and supported
        if "audio" in available_modalities:
            try:
                if INSTRUCTOR_AVAILABLE:
                    if audio_url:
                        from instructor.multimodal import Audio
                        content.append(Audio.from_url(audio_url))
                    elif audio_path:
                        from instructor.multimodal import Audio
                        content.append(Audio.from_path(audio_path))
            except Exception as e:
                logging.warning(f"Failed to load audio: {e}")

        # Return message with multi-modal content
        return [{"role": "user", "content": content}]

    def _generate_single(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single response using instructor.

        Args:
            payload: Input payload containing messages, response_format, etc.

        Returns:
            Dict containing the structured result, token counts, and any errors
        """
        messages = self.build_messages(payload)
        response_format = payload.get("response_format")
        result_key = payload.get("result_key", "result")

        if not response_format:
            raise ValueError("response_format (Pydantic model) is required for InstructorLanguageModel")

        try:
            # Use instructor's structured output
            result = self.client.chat.completions.create(
                messages=messages,
                response_model=response_format,
            )

            # Extract token usage if available
            input_tokens = getattr(result, '_raw_response', {}).get('usage', {}).get('prompt_tokens', 0)
            output_tokens = getattr(result, '_raw_response', {}).get('usage', {}).get('completion_tokens', 0)

            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                result_key: result.model_dump() if hasattr(result, 'model_dump') else result,
            }
        except Exception as e:
            logging.error(f"Instructor generation error: {e}")
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                result_key: None,
                "error": str(e),
            }

    def generate(
        self,
        inputs: List[Dict[str, Any]],
        *,
        max_workers: Optional[int] = None,
        metrics_collector: Optional["LLMMetricsCollector"] = None,
    ) -> Tuple[List[Dict[str, Any]], int, float]:
        """Generate batch responses using instructor.

        Args:
            inputs: List of input payloads
            max_workers: Maximum number of worker threads for parallel processing
            metrics_collector: Optional LLMMetricsCollector to automatically track inference time

        Returns:
            Tuple of (results, effective_max_workers, total_inference_time)
        """
        if not inputs:
            return [], 0, 0.0

        start_time = time.time()

        # Set reasonable defaults for max_workers
        if max_workers is None:
            cpu_count = os.cpu_count()
            max_workers = max(1, (cpu_count or 4) - 1)
        max_workers = min(max_workers, len(inputs), 8)  # Cap at 8 for API rate limits

        results: List[Dict[str, Any]] = [{}] * len(inputs)

        def task(idx: int, item: Dict[str, Any]):
            try:
                res = self._generate_single(item)
                results[idx] = res
            except Exception as e:
                logging.error(f"Instructor inference error for item {idx}: {e}")
                result_key = item.get("result_key", "result")
                results[idx] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    result_key: None,
                    "error": str(e),
                }

        # Execute tasks with progress tracking
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(task, idx, item) for idx, item in enumerate(inputs)]

            # Wait for completion with progress bar
            for _ in as_completed_with_tqdm(futures, desc="Generating"):
                pass

        total_inference_time = time.time() - start_time

        # Automatically add inference time to metrics collector if provided
        if metrics_collector is not None:
            metrics_collector.add_inference_time(total_inference_time)

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
    if provider == "instructor":
        return InstructorLanguageModel(api_key=api_key, base_url=base_url, model=model)
    raise ValueError(f"Unsupported provider: {provider}")


# Register the instructor language model if available
if INSTRUCTOR_AVAILABLE:
    register_language_model("instructor", lambda **kw: InstructorLanguageModel(**kw))

