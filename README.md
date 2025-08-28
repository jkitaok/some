# SOME (Structured Object Media Extraction)

A starter code template for LLM-powered structured object and media extraction with comprehensive metrics and analysis.

## Features

- **Flexible LLM Integration**: Support for OpenAI API, local Ollama, or custom providers
- **Schema-Based Extraction**: Use Pydantic models to define extraction schemas
- **Media Support**: Handle text, images, and multimodal content
- **Comprehensive Metrics**: Automatic analysis of extracted data with detailed statistics
- **Batch Processing**: Efficient parallel processing of multiple items
- **Starter Template**: Ready-to-use foundation for your extraction projects
- **Type-Safe**: Full type hints and validation throughout

## Installation

```bash
pip install -e .
```

## Try It Now

Run the complete example to see structured extraction in action:

```bash
# Option 1: Copy and configure environment file
cp .env.example .env
# Edit .env and add your OpenAI API key

# Option 2: Set environment variable directly
export OPENAI_API_KEY=your_key_here

# Run the example
python -m extraction.examples.generic_extraction.run_my_extraction
```

This will:
- Extract product information from sample text
- Evaluate extraction quality
- Show comprehensive metrics and cost analysis
- Demonstrate the full pipeline with real LLM calls

## Quick Start

### 1. Define Your Schema

```python
from pydantic import BaseModel
from typing import List

class Product(BaseModel):
    name: str
    price: float
    features: List[str]
```

### 2. Create a Prompt Builder

```python
from extraction.prompting import BasePromptBuilder

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
from extraction.inference import get_language_model

# Set up data and model
data = [{"text": "Widget X costs $19.99 with wifi and bluetooth"}]
inputs = [ProductPrompt().build(item) for item in data]

# Run extraction
lm = get_language_model(provider="openai", model="gpt-5-nano")  # or "ollama"
results, workers, timing = lm.generate(inputs)

print(results[0]["product"])  # {"name": "Widget X", "price": 19.99, ...}
```

## LLM Providers

**OpenAI (Recommended)**: Add your API key to `.env` file or `export OPENAI_API_KEY=your_key_here`

**Local Ollama (Free)**: Install from https://ollama.ai, then `ollama pull qwen3:4b-instruct`

**Configuration**: Copy `.env.example` to `.env` and customize your settings

## Examples & Documentation

- **`extraction/examples/generic_extraction/`** - Complete working example (try the command above!)
- **`extraction/examples/custom_llm_provider/`** - Custom provider implementation
- **`extraction/examples/multimodal_extraction/`** - Media and multimodal content handling
- **`docs/DEVELOPER_GUIDE.md`** - Build custom extraction pipelines
- **`docs/SCHEMA_METRICS.md`** - Analyze extracted data quality

## Getting Started with SOME

This template provides everything you need to build structured extraction systems:

1. **Clone and customize** the schemas in `extraction/examples/`
2. **Modify prompts** to match your specific extraction needs
3. **Add your data sources** and run extractions
4. **Analyze results** with built-in metrics and validation
