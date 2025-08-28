# Generic Extraction Example

A streamlined example demonstrating structured data extraction with automatic evaluation using the SOME library.

## 🎯 What This Example Shows

- **Simple JSON Dataset**: Clean dataset format with optional ground truth
- **Product data extraction** from text descriptions
- **Automatic evaluation**: Runs both manual and labeled evaluation when applicable
- **Easy adoption**: Minimal configuration, maximum functionality
- **Performance metrics**: Cost, timing, and success rate tracking

## 🚀 Quick Start

```bash
# Just run it - automatically evaluates based on your data
python -m some.examples.generic_extraction.run_extraction

# Use your own dataset
python -m some.examples.generic_extraction.run_extraction --dataset my_data.json
```

## 🧠 Automatic Evaluation

The system automatically runs appropriate evaluations:

- **Always runs manual evaluation**: General quality assessment
- **Runs labeled evaluation when ground truth is available**: Detailed accuracy analysis
- **No configuration needed**: Just provide data and go

### What You Get

**Manual Evaluation** (always runs):
- ✅ Factual accuracy assessment
- ✅ Schema compliance checking
- ✅ 100% success rate in this example

**Labeled Data Evaluation** (when ground truth provided):
- ✅ Exact match detection (100% in this example)
- ✅ Field-level accuracy analysis
- ✅ Comprehensive metrics

## 📋 Dataset Format

The example uses a simple JSON dataset file (`dataset.json`):

```json
[
  {
    "id": "product_001",
    "text": "Widget X costs $19.99 and includes wifi, gps.",
    "ground_truth": {
      "name": "Widget X",
      "price": 19.99,
      "features": ["wifi", "gps"]
    }
  }
]
```

### Required Fields
- `id`: Unique identifier for the item
- `text`: Text for extraction

### Optional Fields
- `ground_truth`: Expected extraction result (enables labeled evaluation)

### Dataset Types
- **With ground truth**: Automatically runs labeled data evaluation
- **Without ground truth**: Automatically runs manual evaluation

## 🔧 Key Integration: Dual Evaluation System

This example demonstrates both manual and labeled data evaluation:

```python
# Manual evaluation (no ground truth)
manual_results, manual_data = run_manual_evaluation(inputs, results, lm, llm_collector)

# Labeled evaluation (with ground truth)
labeled_results, labeled_data = run_labeled_evaluation(dataset, results, lm, llm_collector)
```

### Integration Benefits
- **Flexible evaluation**: Choose the right evaluation mode for your needs
- **Comprehensive metrics**: Get both qualitative and quantitative assessments
- **Easy integration**: Built on the premade evaluation system
- **Reusable**: Works with any extraction schema and dataset format

## 📁 File Structure

```
generic_extraction/
├── dataset.json           # JSON dataset with labeled ground truth data
├── my_schema.py           # ProductSpec schema definition
├── my_prompt.py           # ProductPrompt builder
├── my_language_model.py   # Custom model registration (optional)
├── run_extraction.py      # Main script with dual evaluation system
└── README.md             # This file
```

## 🔍 Evaluation Results

### Manual Evaluation Results
```
Manual Evaluation Results:
  Item 1: Correct=True, Formatted=True
    Reasoning: The extraction correctly captured name 'Widget X', price 19.99...
  Item 2: Correct=True, Formatted=True
    Reasoning: Extracted name 'Gadget Y', price 49.5, and features...
```

### Labeled Data Evaluation Results
```
Labeled Evaluation Results (10 items):
  Item 1 (product_001):
    Exact Match: ✅
    Partial Match: ❌
    Accuracy Score: 1.00
  Item 4 (product_004):
    Exact Match: ❌
    Partial Match: ✅
    Accuracy Score: 0.67
    Incorrect Fields (1): name

📊 Labeled Evaluation Summary:
  Exact Matches: 8/10 (80.0%)
  Partial Matches: 3/10 (30.0%)
  Average Accuracy: 0.934
```

### Comprehensive Metrics
- **Extraction success rate**: 10/10 (100.0%)
- **Exact match rate**: 80.0% (labeled evaluation)
- **Field-level accuracy**: Per-field analysis
- **Cost and timing**: $0.014042, 39.19s total
- **Token usage**: Detailed input/output token counts

## 🛠 Customization

### Command Line Options

```bash
--dataset DATASET    # Path to JSON dataset file (default: dataset.json)
```

### Using Different Models

```python
# OpenAI (default)
lm = get_language_model(provider="openai", model="gpt-4o-mini")

# Custom model (registered in my_language_model.py)
lm = get_language_model(provider="custom", model="mock-model")

# Ollama local
lm = get_language_model(provider="ollama", model="llama3:8b")
```

### Creating Your Own Dataset

Create a simple JSON file with your data:

```python
import json

dataset = [
    {
        "id": "item_001",
        "text": "Your product description here...",
        "ground_truth": {
            "name": "Expected Product Name",
            "price": 99.99,
            "features": ["feature1", "feature2"]
        }
    },
    {
        "id": "item_002",
        "text": "Another product description..."
        # No ground_truth = manual evaluation only
    }
    # Add more items...
]

with open("my_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)
```

Then run with your dataset:
```bash
python -m some.examples.generic_extraction.run_extraction --dataset my_dataset.json
```

### Adapting for Different Schemas

To use with your own schema, update:

1. **Schema definition** in `my_schema.py`
2. **Prompt builder** in `my_prompt.py`
3. **Dataset ground truth** to match your schema
4. **Evaluation record formatting** if needed

## 📊 Metrics Output

The example provides comprehensive metrics:

### LLM Performance Metrics
- Token usage and costs
- Processing times
- Success rates
- Worker utilization

### Data Quality Metrics  
- Field coverage analysis
- Value distributions
- Completeness assessment
- Evaluation accuracy rates

## 💡 Key Benefits of Premade Evaluation

1. **Minimal Integration**: Just 2-3 lines of code
2. **Consistent Results**: Same evaluation logic across all pipelines
3. **Reusable**: Works with any extraction schema
4. **Simple Output**: Clear boolean results with optional details
5. **Maintainable**: Centralized evaluation logic

## 🔗 Related Examples

- [Premade Templates](../../premade/) - Ready-to-run examples including evaluation
- [Simple Product Extraction](../../premade/simple_product_extraction/) - Basic extraction template
- [Extraction Evaluation](../../premade/extraction_evaluation/) - Standalone evaluation example

---

**Ready to integrate evaluation into your pipeline?** Just import and use! 🎉
