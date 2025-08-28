# Pre-made Examples - Quick Start Templates

Welcome to the SOME library's pre-made examples! These are ready-to-run templates designed to help you get started quickly with common extraction and evaluation tasks.

## 🚀 Quick Start

Each example is completely self-contained and can be run immediately:

```bash
# Run the simple product extraction example
python -m some.premade.simple_product_extraction.run_simple_extraction

# Run the extraction evaluation example
python -m some.premade.extraction_evaluation.run_evaluation
```

## 📁 Available Examples

### 1. Simple Product Data Extraction
**Location:** `simple_product_extraction/`

A complete pipeline for extracting structured product information from text descriptions.

**What it demonstrates:**
- Schema definition with Pydantic models
- Prompt building for extraction tasks
- Language model integration
- Metrics collection and analysis
- Sample data processing

**Key files:**
- `schema.py` - ProductData schema with comprehensive fields
- `prompt.py` - ProductExtractionPrompt builder
- `run_simple_extraction.py` - Complete runnable example

**Sample output:**
```
🚀 Simple Product Data Extraction Example
📝 Processing 5 product descriptions...
🤖 Using openai language model: gpt-4o-mini
⚡ Running extraction...
✅ Extraction completed using 3 workers in 2.45s

📊 EXTRACTION RESULTS
🏷️  Product 1:
   Name: Apple iPhone 15 Pro Max
   Price: $1199.0
   Brand: Apple
   Category: electronics
   Features: 6.7-inch Super Retina XDR display, A17 Pro chip, titanium design...
```

### 2. Extraction Evaluation
**Location:** `extraction_evaluation/`

A simple system for evaluating extraction results against schemas and quality criteria.

**What it demonstrates:**
- Two-criteria evaluation (correctness and formatting)
- Simple boolean assessment with optional reasoning
- Evaluation prompt building
- Basic quality metrics
- Sample evaluation scenarios

**Key files:**
- `schema.py` - BasicEvaluation schema with 4 simple fields
- `prompt.py` - EvaluationPrompt builder
- `run_evaluation.py` - Complete evaluation pipeline

**Sample output:**
```
🔍 Extraction Evaluation Example
📋 Evaluating 4 extraction scenarios...
🤖 Using openai language model: gpt-4o
⚡ Running evaluations...

📊 EVALUATION RESULTS
🎯 Scenario 1: Good Product Extraction
   Correct: ✅ Yes
   Formatted: ✅ Yes
   Confidence: 0.95
   Reasoning: Extraction is factually accurate and follows schema properly
```

## 🛠 Building Blocks Used

Both examples demonstrate the core SOME library components:

### Schema Definition
```python
from pydantic import BaseModel, Field

class MySchema(BaseModel):
    field1: str = Field(description="Description of field")
    field2: Optional[float] = Field(default=None)
```

### Prompt Building
```python
from some.prompting import BasePromptBuilder

class MyPrompt(BasePromptBuilder):
    def build(self, item):
        return {
            "prompt_text": f"Extract data from: {item['text']}",
            "response_format": MySchema,
            "result_key": "extracted_data"
        }
```

### Language Model Integration
```python
from some.inference import get_language_model

lm = get_language_model(provider="openai", model="gpt-4o-mini")
results, workers, time = lm.generate(inputs)
```

### Metrics Collection
```python
from some.metrics import LLMMetricsCollector, SchemaMetricsCollector

# LLM performance metrics
llm_collector = LLMMetricsCollector(name="my_task")
llm_metrics = llm_collector.collect_metrics(results)

# Schema-based data quality metrics
schema_collector = SchemaMetricsCollector(MySchema, "analysis")
schema_metrics = schema_collector.collect_metrics(extracted_data)
```

## 🎯 Customization Guide

### Adapting the Product Extraction Example

1. **Modify the schema** (`schema.py`):
   ```python
   class MyProductSchema(BaseModel):
       # Add your specific fields
       custom_field: str
       industry_specific_data: Optional[Dict[str, Any]]
   ```

2. **Update the prompt** (`prompt.py`):
   ```python
   def build(self, item):
       return {
           "prompt_text": f"Extract {your_specific_requirements}: {item['text']}",
           "response_format": MyProductSchema,
           "result_key": "my_data"
       }
   ```

3. **Replace sample data** in `run_simple_extraction.py`

### Adapting the Evaluation Example

1. **Define evaluation criteria** for your domain
2. **Customize the evaluation prompt** with domain-specific instructions
3. **Add your own test scenarios** with known good/bad examples

## 🔧 Configuration Options

### Language Model Selection
```python
# OpenAI (default)
lm = get_language_model(provider="openai", model="gpt-4o-mini")

# Ollama (local)
lm = get_language_model(provider="ollama", model="llama3:8b")

# Custom provider
lm = get_language_model(provider="custom", model="my-model")
```

### Cost Management
```python
llm_collector = LLMMetricsCollector(
    name="my_extraction",
    cost_per_input_token=0.15/1000000,   # Adjust for your model
    cost_per_output_token=0.6/1000000
)
```

## 📊 Understanding the Output

### LLM Performance Metrics
- **Total items**: Number of inputs processed
- **Success rate**: Percentage of successful extractions
- **Token usage**: Input/output tokens consumed
- **Cost analysis**: Estimated API costs
- **Timing**: Processing speed metrics

### Schema Quality Metrics
- **Field coverage**: Which fields are populated
- **Data types**: Type compliance analysis
- **Value distributions**: Statistical analysis of extracted values
- **Completeness**: Missing data analysis

### Evaluation Metrics
- **Overall scores**: Weighted quality assessment
- **Criterion breakdown**: Performance per evaluation dimension
- **Issue classification**: Severity and type of problems found
- **Recommendations**: Actionable improvement suggestions

## 🚦 Next Steps

1. **Run the examples** to see them in action
2. **Examine the code** to understand the patterns
3. **Customize for your use case** by modifying schemas and prompts
4. **Add your own data** and test with real scenarios
5. **Extend with additional features** like custom metrics or evaluation criteria

## 💡 Tips for Success

- Start with small datasets to test and iterate quickly
- Use cost-effective models (like gpt-4o-mini) for development
- Monitor token usage and costs during development
- Validate your schemas with sample data before large runs
- Use evaluation examples to improve your extraction prompts

## 🔗 Related Documentation

- [Developer Guide](../../docs/DEVELOPER_GUIDE.md) - Building custom pipelines
- [Schema Metrics Guide](../../docs/SCHEMA_METRICS.md) - Advanced metrics analysis
- [Main Examples](../) - More specialized examples (vision, audio, multimodal)

---

**Ready to start?** Pick an example and run it now! 🎉
