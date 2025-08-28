# Simple Product Data Extraction

A ready-to-run example demonstrating product information extraction from text descriptions using the SOME library.

## 🎯 What This Example Does

Extracts structured product information from natural language descriptions, including:
- Product names and brands
- Pricing information
- Feature lists and specifications
- Categories and ratings
- Availability status

## 🚀 Quick Start

```bash
# Run the example
python -m some.premade.simple_product_extraction.run_simple_extraction

# Or import and use in your code
from some.premade.simple_product_extraction import main
results = main()
```

## 📋 Sample Input/Output

**Input text:**
```
"The Apple iPhone 15 Pro Max features a 6.7-inch Super Retina XDR display, 
A17 Pro chip, and titanium design. Starting at $1,199, it includes a 48MP 
main camera, 5x telephoto zoom, and up to 1TB storage."
```

**Extracted output:**
```json
{
  "name": "Apple iPhone 15 Pro Max",
  "price": 1199.0,
  "category": "electronics",
  "brand": "Apple",
  "features": [
    "6.7-inch Super Retina XDR display",
    "A17 Pro chip", 
    "titanium design",
    "48MP main camera",
    "5x telephoto zoom"
  ],
  "description": "Premium smartphone with advanced camera system",
  "model_number": null,
  "rating": null,
  "availability": null
}
```

## 🏗 Architecture

### Files Structure
```
simple_product_extraction/
├── __init__.py          # Package exports
├── schema.py            # ProductData schema definition
├── prompt.py            # ProductExtractionPrompt builder
├── run_simple_extraction.py  # Main execution script
└── README.md           # This file
```

### Key Components

#### 1. Schema (`schema.py`)
- **ProductData**: Main schema with comprehensive product fields
- **ProductCategory**: Enum for product classification
- Field validation and type safety with Pydantic

#### 2. Prompt Builder (`prompt.py`)
- **ProductExtractionPrompt**: Converts text to extraction prompts
- Optimized for various product description formats
- Clear instructions for accurate extraction

#### 3. Execution Script (`run_simple_extraction.py`)
- Sample data with diverse product types
- Complete pipeline demonstration
- Metrics collection and analysis

## 🛠 Customization

### Adding New Product Fields

1. **Extend the schema:**
```python
class ProductData(BaseModel):
    # Existing fields...
    warranty_period: Optional[str] = Field(default=None)
    dimensions: Optional[Dict[str, float]] = Field(default=None)
    weight: Optional[float] = Field(default=None)
```

2. **Update the prompt:**
```python
def build(self, item):
    return {
        "prompt_text": f"""Extract product info including warranty, dimensions, 
        and weight if mentioned: {item['text']}""",
        "response_format": ProductData,
        "result_key": "product_data"
    }
```

### Using Different Language Models

```python
# Cost-effective option
lm = get_language_model(provider="openai", model="gpt-4o-mini")

# Local model
lm = get_language_model(provider="ollama", model="llama3:8b")

# High-performance option
lm = get_language_model(provider="openai", model="gpt-4o")
```

### Processing Your Own Data

Replace the sample data in `get_sample_data()`:

```python
def get_sample_data():
    return [
        {"text": "Your product description 1..."},
        {"text": "Your product description 2..."},
        # Add more items
    ]
```

Or load from file:
```python
from some.io import read_json

def get_sample_data():
    return read_json("your_product_data.json")
```

## 📊 Understanding the Output

### Extraction Results
Each product shows:
- **Name**: Extracted product title
- **Price**: Numerical price value (USD)
- **Brand**: Manufacturer or brand name
- **Category**: Classified product type
- **Features**: List of key product attributes

### Performance Metrics
- **Processing time**: Speed per product
- **Token usage**: API consumption
- **Cost analysis**: Estimated expenses
- **Success rate**: Extraction reliability

### Data Quality Analysis
- **Field coverage**: Which fields are populated
- **Value distributions**: Statistical analysis
- **Completeness**: Missing data patterns

## 🎛 Configuration Options

### Adjusting Extraction Behavior

```python
# More detailed extraction
prompt_text = f"""Extract comprehensive product information including 
technical specifications, compatibility, and user benefits: {text}"""

# Focused extraction
prompt_text = f"""Extract only essential product details - name, price, 
and key features: {text}"""
```

### Cost Management

```python
llm_collector = LLMMetricsCollector(
    name="product_extraction",
    cost_per_input_token=0.15/1000000,   # GPT-4o-mini pricing
    cost_per_output_token=0.6/1000000
)
```

## 🔍 Troubleshooting

### Common Issues

**Empty extractions:**
- Check if input text contains product information
- Verify language model connectivity
- Review prompt clarity

**Incorrect prices:**
- Ensure price format is clear in source text
- Check for currency conversion needs
- Validate price extraction logic

**Missing features:**
- Increase prompt specificity
- Check feature list formatting
- Consider text preprocessing

### Debugging Tips

```python
# Add debug output
for i, result in enumerate(results):
    print(f"Input {i}: {sample_items[i]['text'][:100]}...")
    print(f"Output {i}: {result}")
    print("-" * 50)
```

## 🚀 Next Steps

1. **Test with your data**: Replace sample data with real product descriptions
2. **Customize the schema**: Add fields specific to your domain
3. **Optimize prompts**: Refine extraction instructions for better accuracy
4. **Scale up**: Process larger datasets with batch operations
5. **Add validation**: Implement business rules and data quality checks

## 🔗 Related Examples

- [Extraction Evaluation](../extraction_evaluation/) - Evaluate extraction quality
- [Vision Extraction](../../vision_extraction/) - Extract from product images
- [Multimodal Extraction](../../multimodal_extraction/) - Combine text and images

---

**Ready to extract product data?** Run the example and start customizing! 🛍️
