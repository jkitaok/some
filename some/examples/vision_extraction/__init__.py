"""
Vision Product Extraction Example

This package demonstrates how to use the some library for extracting structured
product information from images using vision-language models.

Key components:
- ProductDetails: Comprehensive schema for product information
- ProductExtractionPrompt: Prompt builder for image-based extraction
- ProductEvaluationPrompt: Prompt builder for quality evaluation
- run_multimodal_extraction: Main execution script

Example usage:
    from some.examples.vision_extraction import main
    results = main()
"""

from .product_schema import ProductDetails, ProductEvaluation, ProductCategory, ProductCondition
from .product_prompt import ProductExtractionPrompt, ProductEvaluationPrompt, ProductComparisonPrompt
from .run_multimodal_extraction import main, load_sample_data, format_evaluation_record

__all__ = [
    "ProductDetails",
    "ProductEvaluation", 
    "ProductCategory",
    "ProductCondition",
    "ProductExtractionPrompt",
    "ProductEvaluationPrompt",
    "ProductComparisonPrompt",
    "main",
    "load_sample_data",
    "format_evaluation_record"
]
