# Vision Product Extraction Example

This example demonstrates how to use the `some` library for extracting structured product information from images using vision-language models. It showcases computer vision AI capabilities for e-commerce, inventory management, and product cataloging applications.

## 🎯 What This Example Does

**Note**: The sample product images referenced in this example were generated using [FLUX.1-dev](https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev), a state-of-the-art text-to-image generation model by Black Forest Labs.

1. **Product Detail Extraction**: Analyzes product images to extract comprehensive information including:
   - Product name, brand, and category
   - Pricing and discount information
   - Physical characteristics (color, size, material)
   - Key features and specifications
   - Visual quality assessment

2. **Quality Evaluation**: Automatically evaluates extraction accuracy by:
   - Comparing results against expected details
   - Assessing completeness and correctness
   - Providing confidence scores and quality ratings

3. **Performance Metrics**: Tracks and reports:
   - Extraction success rates
   - Processing time and costs
   - Schema compliance metrics
   - Overall system performance

## 📁 Project Structure

```
vision_extraction/
├── __init__.py                     # Package initialization
├── README.md                       # This documentation
├── product_schema.py               # Pydantic schemas for product data
├── product_prompt.py               # Prompt builders for extraction tasks
├── run_multimodal_extraction.py   # Main execution script
├── test_extraction.py              # Simple test script
└── input_dataset/                 # Sample data and images
    ├── README.md                   # Dataset documentation
    ├── sample_products.json        # Sample product metadata
    └── images/                     # Product image files (generated with FLUX.1-dev)
        ├── smartphone_box.jpg      # Electronics example
        ├── coffee_bag.jpg          # Food/beverage example
        ├── running_shoes.jpg       # Sports/apparel example
        ├── skincare_bottle.jpg     # Beauty/cosmetics example
        └── book_cover.jpg          # Books/media example
```

## 🚀 Quick Start

### Prerequisites

1. Install the `instructor` library for multimodal support:
   ```bash
   pip install instructor
   ```

2. Set up your API keys (choose one):
   ```bash
   # For OpenAI
   export OPENAI_API_KEY="your-openai-api-key"
   
   # For Anthropic
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   ```

### Running the Example

1. **From the command line**:
   ```bash
   cd some/examples/vision_extraction
   python run_multimodal_extraction.py
   ```

2. **As a Python module**:
   ```python
   from some.examples.vision_extraction import main
   results = main()
   ```

3. **Custom usage**:
   ```python
   from some.examples.vision_extraction import (
       ProductExtractionPrompt, 
       load_sample_data,
       ProductDetails
   )
   from some.inference import get_language_model
   
   # Load your data
   products = load_sample_data()
   
   # Build extraction inputs
   inputs = [ProductExtractionPrompt().build(item) for item in products]
   
   # Run extraction
   lm = get_language_model(provider="instructor", model="openai/gpt-4o-mini")
   results, workers, timing = lm.generate(inputs)
   ```

## 📊 Schema Overview

### ProductDetails
Comprehensive product information schema with fields for:
- **Basic Info**: name, brand, category
- **Pricing**: price, currency, discounts
- **Physical**: color, size, material, condition
- **Features**: key_features, description
- **Visual**: packaging_type, text_visible, logo_visible
- **Quality**: image_quality, confidence_score

### ProductEvaluation
Quality assessment schema for evaluating extraction results:
- **Accuracy**: correct_identification, accurate_details, complete_extraction
- **Quality**: schema_compliance, reasonable_confidence
- **Assessment**: overall_quality, missing_details, incorrect_details
- **Feedback**: reasoning, suggestions

## 🔧 Customization

### Adding Your Own Images

1. Place image files in `input_dataset/images/`
2. Update `input_dataset/sample_products.json` with metadata:
   ```json
   {
     "id": "your_product_id",
     "image_path": "input_dataset/images/your_image.jpg",
     "additional_text": "Optional context",
     "expected_details": {
       "name": "Expected product name",
       "brand": "Expected brand",
       "category": "electronics"
     }
   }
   ```

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)
- GIF (.gif)

### Model Options

The example supports multiple vision-language models:

```python
# OpenAI GPT-4 Vision
lm = get_language_model(provider="instructor", model="openai/gpt-4o-mini")

# Anthropic Claude Vision
lm = get_language_model(provider="instructor", model="anthropic/claude-3-haiku-20240307")

# Or use the built-in OpenAI provider
lm = get_language_model(provider="openai", model="gpt-4o-mini")
```

## 📈 Performance Optimization

### Batch Processing
The example automatically handles batch processing with configurable parallelism:
- Default: CPU cores - 1
- Maximum: 8 concurrent requests (API rate limit friendly)
- Customizable via `max_workers` parameter

### Cost Management
- Uses cost-effective models (gpt-4o-mini) by default
- Tracks token usage and costs
- Provides detailed cost breakdowns

### Quality Assurance
- Confidence scoring for each extraction
- Automatic quality evaluation
- Schema validation and compliance checking

## 🛠️ Troubleshooting

### Common Issues

1. **"instructor library not found"**
   ```bash
   pip install instructor
   ```

2. **"API key not configured"**
   - Set environment variables for your chosen provider
   - Verify API key permissions for vision models

3. **"Image not found"**
   - Check image paths in sample_products.json
   - Ensure images exist in input_dataset/images/

4. **"No successful extractions"**
   - Verify image quality and readability
   - Check API quotas and rate limits
   - Try with simpler test images first

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🎯 Use Cases

This example is ideal for:

- **E-commerce**: Automated product catalog creation
- **Inventory Management**: Digital asset cataloging
- **Market Research**: Competitive product analysis
- **Quality Control**: Product information verification
- **Data Migration**: Legacy catalog digitization

## 🔄 Integration

The multimodal extraction components can be easily integrated into larger systems:

```python
# Custom extraction pipeline
from some.examples.multimodal_extraction import ProductExtractionPrompt
from your_app import process_product_image

def extract_product_info(image_path: str) -> dict:
    prompt_builder = ProductExtractionPrompt()
    input_data = prompt_builder.build({"image_path": image_path})
    
    # Use your preferred language model
    lm = get_language_model(provider="instructor")
    results, _, _ = lm.generate([input_data])
    
    return results[0].get("product_details", {})
```

## 📚 Further Reading

- [Instructor Library Documentation](https://python.useinstructor.com/)
- [OpenAI Vision API Guide](https://platform.openai.com/docs/guides/vision)
- [Pydantic Schema Documentation](https://docs.pydantic.dev/)
- [Some Library Documentation](../../README.md)
