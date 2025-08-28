"""
Simple Product Data Extraction - Pre-made Example

This package demonstrates a ready-to-use product data extraction pipeline
using the SOME library's building blocks.

Key components:
- ProductData: Schema for structured product information
- ProductExtractionPrompt: Prompt builder for product data extraction
- run_simple_extraction: Main execution script with sample data

Example usage:
    from some.premade.simple_product_extraction import main
    results = main()

    # Or run directly
    python -m some.premade.simple_product_extraction.run_simple_extraction
"""

from .schema import ProductData
from .prompt import ProductExtractionPrompt
from .run_simple_extraction import main

__all__ = [
    "ProductData",
    "ProductExtractionPrompt", 
    "main"
]
