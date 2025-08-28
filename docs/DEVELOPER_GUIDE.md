# Developer Guide: SOME Template

Build custom LLM-powered structured object and media extraction pipelines using this starter template.

## Core Components

- **BaseLanguageModel** (`some/inference.py`): Handles batch generation and provider abstraction
- **BasePromptBuilder** (`some/prompting.py`): Builds model inputs with:
  - `messages`: Chat messages for the LLM
  - `response_format`: Pydantic model for structured output
  - `result_key`: Key name for storing results

## Basic Workflow

### 1. Define Schema

```python
from pydantic import BaseModel
from typing import List

class Product(BaseModel):
    name: str
    price: float
    features: List[str]
```

### 2. Create Prompt Builder

```python
from some.prompting import BasePromptBuilder

class ProductPrompt(BasePromptBuilder):
    def build(self, item):
        return {
            "messages": [{"role": "user", "content": f"Extract product info: {item['text']}"}],
            "response_format": Product,
            "result_key": "product"
        }
```

### 3. Run Extraction

```python
from some.inference import get_language_model

# Prepare data
items = [{"text": "Widget X costs $19.99 with wifi and GPS"}]
inputs = [ProductPrompt().build(item) for item in items]

# Extract
lm = get_language_model(provider="openai")
results, workers, timing = lm.generate(inputs)
print(results[0]["product"])
```

## Advanced Features

### Custom Providers
```python
from some.inference import BaseLanguageModel, register_language_model

class CustomLM(BaseLanguageModel):
    def generate(self, inputs, *, max_workers=None):
        return results, workers, timing

register_language_model("custom", lambda **kw: CustomLM(**kw))
```

### Data Analysis
```python
from some.metrics import SchemaMetricsCollector

collector = SchemaMetricsCollector(Product, "analysis")
metrics = collector.collect_metrics(extracted_data)
print(collector.format_summary(metrics))
```

### File I/O Utilities
```python
from some.main import load_extraction_data
from some.io import write_json

# Load data from JSON, JSONL, or TXT
data = load_extraction_data("input.json")

# Save results
write_json("results.json", extracted_data)
```

## Try the Example

```bash
python -m some.examples.generic_extraction.run_my_extraction
```

This demonstrates the complete pipeline with real LLM calls, metrics, and evaluation.

## Testing

Run tests: `cd tests && python run_all_tests.py`
