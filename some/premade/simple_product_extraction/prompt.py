"""
Product Extraction Prompt Builder

Defines the prompt template for extracting structured product data from text.
Uses the BasePromptBuilder interface from the SOME library.
"""
from __future__ import annotations

from typing import Any, Dict

from .schema import ProductData
from some.prompting import BasePromptBuilder

class ProductExtractionPrompt(BasePromptBuilder):
    """
    Prompt builder for extracting product information from text descriptions.
    
    This prompt is designed to work with various types of product-related text:
    - Product descriptions
    - Marketing copy
    - Customer reviews
    - Product listings
    - Specification sheets
    """
    
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a prompt for product data extraction.
        
        Args:
            item: Dictionary containing 'text' key with product description
            
        Returns:
            Dictionary with prompt configuration for the language model
        """
        text = item["text"]
        
        prompt_text = f"""Extract structured product information from the following text. 
Focus on identifying key product details like name, price, features, brand, and other relevant information.

If certain information is not mentioned or unclear, leave those fields empty or null.
Be accurate and only extract information that is explicitly stated or can be reasonably inferred.

Text to analyze:
{text}

Please extract the product information in the specified JSON format."""

        return {
            "prompt_text": prompt_text,
            "response_format": ProductData,
            "result_key": "product_data",
        }
