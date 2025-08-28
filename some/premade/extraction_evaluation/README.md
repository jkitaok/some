# Extraction Evaluation

A comprehensive system for evaluating extraction results against schemas and quality criteria using the SOME library. Supports both manual evaluation and labeled data evaluation modes.

## 🎯 What This Example Does

### Manual Evaluation Mode
Evaluates extraction quality on two key criteria:
- **Correct**: Is the extracted information factually accurate?
- **Formatted**: Does the output follow the expected schema properly?

### Labeled Data Evaluation Mode
Provides detailed comparison against ground truth data:
- **Exact Match**: Does the extraction exactly match the ground truth?
- **Partial Match**: Does the extraction partially match the ground truth?
- **Field-Level Analysis**: Per-field accuracy assessment
- **Quantitative Metrics**: Accuracy, precision, recall, and F1 scores
- **Missing/Extra/Incorrect Fields**: Detailed field-level analysis

## 🚀 Quick Start

### Manual Evaluation
```bash
# Run manual evaluation with sample scenarios
python -m some.premade.extraction_evaluation.run_evaluation

# Or import and use in your code
from some.premade.extraction_evaluation import main
results = main()
```

### Labeled Data Evaluation
```bash
# Run evaluation with labeled ground truth data
python -m some.premade.extraction_evaluation.run_evaluation --labeled-data path/to/labeled_data.json

# Or import and use in your code
from some.premade.extraction_evaluation import main
results = main(labeled_data_path="path/to/labeled_data.json")
```

## 📋 Labeled Data Format

### Required Fields
Each labeled data item must contain:
```json
{
  "id": "unique_identifier",
  "original_text": "The source text that was processed for extraction",
  "ground_truth": {
    "field1": "expected_value1",
    "field2": ["expected", "list", "values"],
    "field3": 42
  }
}
```

### Optional Fields
```json
{
  "extraction_prompt": "The prompt used for extraction",
  "expected_schema": {"field1": "string", "field2": "array"},
  "evaluation_context": "Additional context for evaluation"
}
```

### Sample Labeled Data
See `sample_labeled_data.json` for complete examples including:
- Product information extraction
- Book metadata extraction
- Service specification extraction
- Event details extraction

## 📋 Sample Manual Evaluation Scenario

**Original Text:**
```
"Tesla Model 3 Long Range offers up to 358 miles of EPA-estimated range. 
It accelerates from 0-60 mph in 4.2 seconds and has a top speed of 145 mph. 
Starting price is $47,740."
```

**Expected Schema:**
```json
{
  "model": "string",
  "range_miles": "number", 
  "acceleration_0_60": "number",
  "price": "number"
}
```

**Extraction Output:**
```json
{
  "model": "Tesla Model 3 Long Range",
  "range_miles": 400,  // ❌ Incorrect: should be 358
  "acceleration_0_60": 3.8,  // ❌ Incorrect: should be 4.2  
  "price": 47740  // ✅ Correct
}
```

**Evaluation Result:**
```json
{
  "correct": false,
  "formatted": true,
  "reasoning": "Extraction contains factual inaccuracies: range should be 358 miles, not 400; acceleration should be 4.2 seconds, not 3.8",
  "confidence": 0.92
}
```

## 🏗 Architecture

### Files Structure
```
extraction_evaluation/
├── __init__.py              # Package exports
├── schema.py                # Evaluation schemas (Basic & Labeled)
├── prompt.py                # Prompt builders for both modes
├── labeled_utils.py         # Utilities for labeled data handling
├── run_evaluation.py        # Main execution supporting both modes
├── sample_labeled_data.json # Sample labeled data for testing
└── README.md               # This file
```

### Key Components

#### 1. Schemas (`schema.py`)
- **BasicEvaluation**: Simple evaluation for manual mode
  - `correct`: Boolean indicating factual accuracy
  - `formatted`: Boolean indicating schema compliance
  - `reasoning`: Optional explanation of the evaluation
  - `confidence`: Optional confidence score (0.0-1.0)

- **LabeledEvaluation**: Enhanced evaluation for labeled data mode
  - `exact_match`: Boolean for perfect match with ground truth
  - `partial_match`: Boolean for partial match
  - `field_accuracy`: Per-field accuracy mapping
  - `missing_fields`: Fields in ground truth but not extraction
  - `extra_fields`: Fields in extraction but not ground truth
  - `incorrect_fields`: Fields with wrong values
  - `accuracy_score`: Overall accuracy (0.0-1.0)
  - `precision`, `recall`, `f1_score`: Metrics for multi-value fields

#### 2. Prompt Builders (`prompt.py`)
- **EvaluationPrompt**: Manual evaluation prompts
- **LabeledEvaluationPrompt**: Ground truth comparison prompts
- Both focus on clear, detailed evaluation instructions

#### 3. Labeled Data Utilities (`labeled_utils.py`)
- **load_labeled_data()**: Load and validate labeled data files
- **calculate_accuracy_metrics()**: Compute quantitative metrics
- **analyze_field_differences()**: Identify missing/extra/incorrect fields
- **format_evaluation_record()**: Prepare data for evaluation

#### 4. Execution Script (`run_evaluation.py`)
- Supports both manual and labeled evaluation modes
- Command-line interface with argument parsing
- Complete evaluation pipeline with metrics collection
- Detailed results analysis and reporting

## 🔧 Integration with Your Pipeline

### Using Labeled Data Evaluation in Your Code

```python
from some.premade.extraction_evaluation import (
    LabeledEvaluationPrompt,
    load_labeled_data,
    format_evaluation_record
)
from some.inference import get_language_model

# Load your labeled data
labeled_data = load_labeled_data("your_labeled_data.json")

# Run your extraction pipeline on the original texts
extraction_results = []
for item in labeled_data:
    # Your extraction logic here
    result = your_extraction_pipeline(item["original_text"])
    extraction_results.append(result)

# Evaluate against ground truth
evaluation_records = [
    format_evaluation_record(item, result)
    for item, result in zip(labeled_data, extraction_results)
]

# Build evaluation prompts
prompt_builder = LabeledEvaluationPrompt()
inputs = [prompt_builder.build(record) for record in evaluation_records]

# Run evaluations
lm = get_language_model(provider="openai", model="gpt-4o")
results, _, _ = lm.generate(inputs)
```

### Creating Your Own Labeled Data

1. **Prepare your data in the required format:**
```python
labeled_data = [
    {
        "id": "item_001",
        "original_text": "Your source text here...",
        "ground_truth": {
            "field1": "expected_value",
            "field2": ["list", "of", "values"]
        },
        "extraction_prompt": "Your extraction prompt",
        "expected_schema": {"field1": "string", "field2": "array"}
    }
]
```

2. **Save as JSON or JSONL:**
```python
from some.io import write_json
write_json("my_labeled_data.json", labeled_data)
```

3. **Run evaluation:**
```bash
python -m some.premade.extraction_evaluation.run_evaluation --labeled-data my_labeled_data.json
```

## 🛠 Customization

### Adding Custom Evaluation Criteria

1. **Extend the evaluation schemas:**
```python
from some.premade.extraction_evaluation.schema import LabeledEvaluation
from pydantic import BaseModel, Field

class CustomLabeledEvaluation(LabeledEvaluation):
    domain_specificity: bool = Field(description="Is the extraction domain-appropriate?")
    business_logic: bool = Field(description="Does it follow business rules?")
    custom_score: float = Field(ge=0.0, le=1.0, description="Custom scoring metric")
```

2. **Create custom prompt builders:**
```python
from some.premade.extraction_evaluation.prompt import LabeledEvaluationPrompt

class CustomLabeledEvaluationPrompt(LabeledEvaluationPrompt):
    def build(self, item):
        base_prompt = super().build(item)
        # Add custom evaluation instructions
        base_prompt["prompt_text"] += "\n\nAdditional custom criteria:\n- Domain specificity\n- Business logic compliance"
        base_prompt["response_format"] = CustomLabeledEvaluation
        return base_prompt
```

### Metrics and Analysis

The system provides comprehensive metrics:

- **Accuracy Metrics**: Overall accuracy, field-level accuracy
- **Precision/Recall/F1**: For multi-value fields (lists, arrays)
- **Field Analysis**: Missing, extra, and incorrect fields
- **Aggregate Statistics**: Across all evaluation items

### Performance Considerations

- **Model Choice**: Use GPT-4o or similar capable models for accurate evaluation
- **Batch Processing**: Process multiple evaluations in parallel
- **Cost Optimization**: Consider using smaller models for simpler evaluations

## 🧪 Testing

### Run with Sample Data
```bash
# Test manual evaluation
python -m some.premade.extraction_evaluation.run_evaluation

# Test labeled data evaluation
python -m some.premade.extraction_evaluation.run_evaluation --labeled-data some/premade/extraction_evaluation/sample_labeled_data.json
```

### Verify Your Labeled Data
```python
from some.premade.extraction_evaluation.labeled_utils import load_labeled_data, validate_labeled_item

# Load and validate your data
try:
    data = load_labeled_data("your_data.json")
    for item in data:
        validate_labeled_item(item)
    print(f"✅ Successfully validated {len(data)} items")
except Exception as e:
    print(f"❌ Validation error: {e}")
```

## 📊 Output Examples

### Manual Evaluation Output
```
📊 MANUAL EVALUATION RESULTS
==================================================

📝 Evaluation 1:
   Correct: ✅
   Formatted: ✅
   Reasoning: The extraction correctly identifies all product details...
   Confidence: 0.95
```

### Labeled Data Evaluation Output
```
📊 LABELED EVALUATION RESULTS
==================================================

📝 Evaluation 1:
   Exact Match: ❌
   Partial Match: ✅
   Formatted: ✅
   Accuracy Score: 0.83
   Missing Fields: storage_type
   Incorrect Fields: price
   Reasoning: Most fields extracted correctly, but price was $999 instead of $1099...

📊 AGGREGATE METRICS
==================================================
Exact Matches: 2/5 (40.0%)
Partial Matches: 4/5 (80.0%)
Average Accuracy Score: 0.756
```

## 🤝 Contributing

To extend this evaluation system:

1. **Add new evaluation schemas** in `schema.py`
2. **Create custom prompt builders** in `prompt.py`
3. **Add utility functions** in `labeled_utils.py`
4. **Update exports** in `__init__.py`
5. **Add tests** and documentation

## 📚 Related Examples

- **Generic Extraction**: Basic extraction pipeline setup
- **Vision Extraction**: Multi-modal extraction with images
- **Audio Extraction**: Audio content extraction and analysis

The evaluation system integrates seamlessly with all extraction examples in the SOME library.

### Creating Domain-Specific Evaluations

```python
def build_medical_evaluation_prompt(item):
    """Evaluation prompt for medical data extraction."""
    return {
        "prompt_text": f"""Evaluate medical data extraction with focus on:
        - Clinical accuracy and terminology
        - Patient safety implications  
        - Regulatory compliance
        - Data sensitivity handling
        
        Original text: {item['original_text']}
        Extraction: {item['extraction_output']}
        """,
        "response_format": ExtractionEvaluation,
        "result_key": "medical_evaluation"
    }
```

### Adding Your Own Test Scenarios

```python
def get_custom_scenarios():
    return [
        {
            "scenario_name": "Your Test Case",
            "original_text": "Source text to evaluate...",
            "extraction_prompt": "Prompt used for extraction...",
            "expected_schema": {"field": "type"},
            "extraction_output": {"field": "extracted_value"},
            "evaluation_context": "Additional context for evaluation"
        }
    ]
```

## 📊 Understanding Evaluation Results

### Basic Assessment
- **Correct**: Boolean indicating if extracted information is factually accurate
- **Formatted**: Boolean indicating if output follows the expected schema
- **Reasoning**: Optional explanation of the evaluation decision
- **Confidence**: Optional confidence score from 0.0 to 1.0

### Quality Analysis
- **Correctness Rate**: Percentage of extractions that are factually accurate
- **Format Compliance**: Percentage of extractions that follow schema properly
- **Average Confidence**: Mean confidence across evaluations with confidence scores

## 🎛 Configuration Options

### Evaluation Sensitivity

```python
# Strict evaluation
prompt_text = """Be very strict in evaluation. Mark as unacceptable 
if any factual errors or format violations exist."""

# Lenient evaluation  
prompt_text = """Focus on major issues. Minor formatting problems 
are acceptable if core information is correct."""
```

### Custom Scoring Weights

```python
# Modify the evaluation prompt to emphasize certain criteria
prompt_text = f"""Weight the criteria as follows:
- Accuracy: 40% (most important)
- Completeness: 30% 
- Format Compliance: 20%
- Consistency: 10%
"""
```

## 🔍 Use Cases

### 1. Quality Assurance
Evaluate extraction pipelines before production deployment:
```python
# Test your extraction system
extraction_results = your_extraction_pipeline(test_data)
evaluation_results = evaluate_extractions(extraction_results)
quality_score = calculate_average_score(evaluation_results)
```

### 2. Model Comparison
Compare different language models or prompts:
```python
models = ["gpt-4o-mini", "gpt-4o", "claude-3-haiku"]
for model in models:
    results = run_extraction_with_model(model, test_data)
    evaluations = evaluate_results(results)
    print(f"{model}: {average_score(evaluations):.2f}")
```

### 3. Continuous Monitoring
Monitor extraction quality over time:
```python
# Daily quality check
daily_extractions = get_recent_extractions()
daily_evaluations = evaluate_batch(daily_extractions)
if average_score(daily_evaluations) < threshold:
    alert_quality_team()
```

## 🚨 Common Evaluation Patterns

### High Accuracy, Low Completeness
- Extraction is correct but misses information
- **Fix**: Improve prompt to request comprehensive extraction

### High Completeness, Low Accuracy  
- Extraction captures everything but with errors
- **Fix**: Add validation steps or use more capable model

### Format Violations
- Correct information in wrong structure
- **Fix**: Improve schema documentation in prompts

### Inconsistent Results
- Similar inputs produce different output formats
- **Fix**: Add examples and stricter formatting instructions

## 🔧 Advanced Features

### Batch Evaluation
```python
def evaluate_batch(extraction_results, batch_size=10):
    """Evaluate multiple extractions efficiently."""
    evaluations = []
    for batch in chunk_list(extraction_results, batch_size):
        batch_evaluations = run_evaluation_batch(batch)
        evaluations.extend(batch_evaluations)
    return evaluations
```

### Custom Issue Detection
```python
def detect_custom_issues(extraction, expected_schema):
    """Add domain-specific issue detection."""
    issues = []
    
    # Example: Check for required business fields
    if extraction.get('price') and extraction['price'] <= 0:
        issues.append({
            "criterion": "business_logic",
            "severity": "critical",
            "description": "Price must be positive"
        })
    
    return issues
```

## 🚀 Next Steps

1. **Run the example**: See evaluation in action with test scenarios
2. **Add your scenarios**: Create test cases from your domain
3. **Customize criteria**: Define evaluation dimensions for your use case
4. **Integrate with pipelines**: Add evaluation to your extraction workflows
5. **Monitor quality**: Set up continuous evaluation for production systems

## 🔗 Related Examples

- [Simple Product Extraction](../simple_product_extraction/) - Generate data to evaluate
- [Generic Extraction](../../generic_extraction/) - More extraction examples
- [Schema Metrics Guide](../../../docs/SCHEMA_METRICS.md) - Advanced analysis

---

**Ready to evaluate extraction quality?** Run the example and start improving! 🔍
