# SOME (Structured Object Media Extraction)

A professional LLM framework for structured extraction from text, images, audio, and multimodal content with built-in evaluation and comprehensive metrics.

## ✨ Key Features

- **🎯 Multimodal Support**: Extract from text, images, audio, and combined modalities
- **🏗️ Modular Architecture**: Reusable `Prompt`, `LanguageModel`, `Schema`, and `Metrics` components
- **📊 Built-in Evaluation**: Same constructs power both extraction and quality assessment
- **🔧 Flexible LLM Integration**: OpenAI, Ollama, Instructor, or custom providers
- **📈 Comprehensive Metrics**: Token usage, costs, timing, and schema-based analysis
- **⚡ Production Ready**: Type-safe, batch processing, error handling

## Installation

```bash
pip install -e .
```

## 🚀 Quick Demo

Experience multimodal extraction in 30 seconds:

```bash
# Set up API key
export OPENAI_API_KEY=your_key_here

# Run complete extraction + evaluation pipeline
python -m some.examples.generic_extraction.run_extraction
```

**What you'll see:**
- 📄 Product extraction from text using structured schemas
- 🔍 Automatic quality evaluation using the same building blocks
- 📊 Comprehensive metrics: tokens, costs, timing, schema analysis
- 🎯 End-to-end pipeline demonstrating all core constructs

## 🏗️ Core Building Blocks

SOME provides four key constructs that work together for both extraction and evaluation:

### 1. Schema - Define Structure
```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD")
    features: List[str] = Field(description="Key features")

# Works for any domain - multimodal analysis, evaluations, etc.
class MultiModalAnalysis(BaseModel):
    content_type: str
    visual_description: Optional[str] = None
    audio_transcript: Optional[str] = None
    confidence_score: float = Field(ge=0.0, le=1.0)
```

### 2. Prompt - Build Requests
```python
from some.prompting import BasePromptBuilder

class ProductPrompt(BasePromptBuilder):
    def build(self, item):
        return {
            "prompt_text": f"Extract product info: {item['text']}",
            "image_path": item.get("image_path"),  # Multimodal support
            "response_format": Product,
            "result_key": "product"
        }

# Same pattern for evaluation prompts
class EvaluationPrompt(BasePromptBuilder):
    def build(self, item):
        return {
            "prompt_text": f"Evaluate extraction quality: {item['extraction']}",
            "response_format": BasicEvaluation,
            "result_key": "evaluation"
        }
```

### 3. LanguageModel - Execute Requests
```python
from some.inference import get_language_model

# Supports text, vision, and multimodal models
lm = get_language_model(provider="openai", model="gpt-4o")  # Vision + text
# lm = get_language_model(provider="ollama", model="llama3:8b")  # Local

# Batch processing with automatic metrics
results, workers, timing = lm.generate(inputs, metrics_collector=collector)
```

### 4. Metrics - Analyze Results
```python
from some.metrics import LLMMetricsCollector, SchemaMetricsCollector

# LLM performance metrics
llm_collector = LLMMetricsCollector(name="extraction")
llm_metrics = llm_collector.collect_metrics(results)
# → tokens, costs, timing, success rates

# Schema-based data quality metrics
schema_collector = SchemaMetricsCollector(Product, "analysis")
schema_metrics = schema_collector.collect_metrics(extracted_data)
# → field completeness, type validation, statistical analysis
```

## 🎭 Multimodal Capabilities

### Text + Vision + Audio
```python
# Analyze presentation slides with speaker audio
data = {
    "text": "Q3 Financial Results",
    "image_path": "slides/q3_chart.png",
    "audio_path": "audio/presentation.wav"
}

# Single prompt handles all modalities
prompt = MultiModalPrompt().build(data)
results = lm.generate([prompt])

# Rich analysis combining all inputs
analysis = results[0]["analysis"]
print(f"Content: {analysis.content_type}")
print(f"Visual: {analysis.visual_description}")
print(f"Audio: {analysis.audio_transcript}")
print(f"Alignment: {analysis.modality_alignment}")
```

### Automatic Modality Detection
```python
# Framework automatically detects available modalities
modalities = ["text", "vision", "audio"]
prompt_builder = determine_prompt_builder(modalities)

# Uses appropriate schema and processing
if "vision" in modalities:
    schema = VisionAudioAnalysis
elif "audio" in modalities:
    schema = TextAudioAnalysis
else:
    schema = TextOnlyAnalysis
```

## 🔍 Evaluation as a First-Class Feature

The same building blocks power both extraction and evaluation:

### Basic Quality Assessment
```python
from some.premade.extraction_evaluation import BasicEvaluation, EvaluationPrompt

# Evaluate any extraction using the same constructs
evaluation_data = {
    "extraction": extracted_product,
    "original_text": source_text,
    "schema": Product
}

# Same prompt builder pattern
eval_prompt = EvaluationPrompt().build(evaluation_data)
eval_results = lm.generate([eval_prompt])

assessment = eval_results[0]["evaluation"]
print(f"Correct: {assessment.correct}")
print(f"Reasoning: {assessment.reasoning}")
```

### Labeled Data Evaluation
```python
from some.premade.extraction_evaluation import LabeledEvaluation

# Compare against ground truth
labeled_eval = {
    "extraction": extracted_data,
    "ground_truth": expected_data,
    "schema": MySchema
}

# Detailed metrics automatically calculated
results = evaluate_against_labels(labeled_eval)
print(f"Exact match: {results.exact_match}")
print(f"Accuracy: {results.accuracy_score}")
print(f"Missing fields: {results.missing_fields_count}")
```

## 🚀 Examples & LLM Providers

### Ready-to-Run Examples
- **`some/premade/`** - Production templates (extraction + evaluation)
- **`some/examples/multimodal_extraction/`** - Text, vision, audio processing
- **`some/examples/generic_extraction/`** - Complete pipeline demo

### LLM Provider Support
```bash
# OpenAI (recommended for multimodal)
export OPENAI_API_KEY=your_key_here

# Local Ollama (free, vision support)
ollama pull qwen3:4b-instruct

# Instructor (structured output)
pip install instructor
```

## 🎯 Why SOME?

**Unified Architecture**: The same `Prompt`, `LanguageModel`, `Schema`, and `Metrics` constructs power extraction, evaluation, and analysis. No separate frameworks needed.

**Multimodal by Design**: Handle text, images, audio, and combinations seamlessly. Automatic modality detection and cross-modal insights.

**Production Ready**: Built-in error handling, batch processing, comprehensive metrics, and cost tracking. Type-safe throughout.

**Evaluation Built-In**: Quality assessment uses the same building blocks as extraction. Easy to validate and improve your pipelines.

---

**Get started:** `python -m some.examples.generic_extraction.run_extraction`

**Documentation:** `docs/DEVELOPER_GUIDE.md` | **Schema Analysis:** `docs/SCHEMA_METRICS.md`
