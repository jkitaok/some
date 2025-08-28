# Generic Extraction with Basic Evaluation

A complete example demonstrating structured data extraction with basic evaluation using the SOME library.

## 🎯 What This Example Shows

- **Simple JSON Dataset**: Clean dataset format without ground truth
- **Product data extraction** from text descriptions
- **Basic evaluation**: Uses premade `BasicEvaluation` structure for quality assessment
- **Automatic metrics collection**: Uses the new `metrics_collector` parameter
- **Performance tracking**: Cost, timing, and success rate metrics

## 🚀 Quick Start

```bash
# Run extraction with basic evaluation
python -m some.examples.extraction_with_basic_evaluation.run_extraction

# Use your own dataset
python -m some.examples.extraction_with_basic_evaluation.run_extraction --dataset my_data.json
```

## 🧠 Basic Evaluation

The system automatically runs basic evaluation using the premade `BasicEvaluation` structure:

- ✅ **Factual accuracy assessment**: Checks if extracted information is correct
- ✅ **Schema compliance checking**: Verifies proper JSON structure and types
- ✅ **Detailed reasoning**: Provides explanations for evaluation decisions
- ✅ **Summary statistics**: Shows overall correctness and formatting rates

### What You Get

**Extraction Results**:
- ✅ 3/3 successful extractions
- ✅ Clean product specifications with name, price, and features

**Basic Evaluation Results**:
- ✅ 100% correct information
- ✅ 100% properly formatted
- ✅ Detailed reasoning for each evaluation

## 📋 Dataset Format

Simple JSON dataset without ground truth:

```json
[
  {
    "id": "product_001",
    "text": "Widget X costs $19.99 and includes wifi, gps."
  },
  {
    "id": "product_002", 
    "text": "Gadget Y is priced at $49.50, features: bluetooth, waterproofing"
  }
]
```

### Required Fields
- `id`: Unique identifier for the item
- `text`: Text for extraction

## 🔧 Key Features

### Automatic Metrics Collection
Uses the new `metrics_collector` parameter for automatic timing:
```python
results, workers, time = lm.generate(inputs, metrics_collector=llm_collector)
# No manual llm_collector.add_inference_time() needed!
```

### Premade Evaluation Structure
Uses the built-in `BasicEvaluation` and `EvaluationPrompt`:
```python
from some.premade.extraction_evaluation import BasicEvaluation, EvaluationPrompt

# Format evaluation records
evaluation_records = [{
    "original_text": input_data["prompt_text"],
    "extraction_prompt": "Extract ProductSpec as JSON...",
    "expected_schema": input_data["response_format"].model_json_schema(),
    "extraction_output": result_data["product"]
}]

# Build evaluation prompts
evaluation_inputs = [EvaluationPrompt().build(record) for record in evaluation_records]
```

## 📊 Example Results

```
🚀 Generic Product Extraction with Basic Evaluation
📋 Loaded 3 items from dataset.json
✅ Successful extractions: 3/3

🔍 Running Basic Evaluation
Basic Evaluation Results (3 items):
  Item 1: Correct=✅, Formatted=✅
  Item 2: Correct=✅, Formatted=✅
  Item 3: Correct=✅, Formatted=✅

📊 Basic Evaluation Summary:
  Correct: 3/3 (100.0%)
  Formatted: 3/3 (100.0%)

💡 SUMMARY
Dataset items: 3
Successful extractions: 3/3 (100.0%)
Total LLM calls: 6 (3 extraction + 3 evaluation)
Total cost: $0.000320
Total time: 2.10s
```

## 🎯 Use Cases

Perfect for:
- **Quality assessment** without ground truth data
- **General evaluation** of extraction accuracy
- **Schema validation** and format checking
- **Development and testing** of extraction pipelines
- **Quick quality checks** on new datasets

## 🔗 Related Examples

- **`generic_extraction`**: Simple extraction only (no evaluation)
- **`extraction_with_labeled_evaluation`**: Extraction with ground truth comparison
- **`multimodal_extraction`**: Image and text extraction
- **`vision_extraction`**: Image-only extraction
