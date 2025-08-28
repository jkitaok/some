from __future__ import annotations

from typing import Any, Dict
import os

from .product_schema import ProductDetails, ProductEvaluation
from some.prompting import BasePromptBuilder
from some.media import get_image_info, is_valid_media_url

class ProductExtractionPrompt(BasePromptBuilder):
    """Prompt builder for extracting product details from images."""
    
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Build prompt for product extraction from image.
        
        Args:
            item: Dict containing:
                - image_path: str - Path to product image
                - text: Optional[str] - Any text context
        """
        image_path = item.get("image_path")
        text = item.get("text", "")

        # Validate image source using media.py functions
        if not image_path:
            raise ValueError("image_path is required for product extraction")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Get image info for better prompting
        try:
            image_info = get_image_info(image_path)
            dimensions = f"{image_info.get('width', 'unknown')}x{image_info.get('height', 'unknown')}"
            text += f"\nImage dimensions: {dimensions}"
        except Exception:
            pass  # Continue without image info if it fails
        
        # Build comprehensive prompt
        prompt_text = """Analyze this product image and extract detailed information following the ProductDetails schema.

EXTRACTION GUIDELINES:
1. **Accuracy First**: Only extract information that is clearly visible or can be reasonably inferred
2. **Confidence Assessment**: Provide an honest confidence score (0.0-1.0) based on image quality and clarity
3. **Complete Analysis**: Look for all visible details including text, logos, colors, materials, and features
4. **Category Classification**: Classify the product into the most appropriate category
5. **Price Information**: Extract any visible pricing, including original prices and discounts

WHAT TO LOOK FOR:
- Product name/title (on packaging, labels, or product itself)
- Brand name and logos
- Price tags, stickers, or printed prices
- Color, size, material information
- Key features or specifications listed
- Packaging type and condition
- Any promotional text or descriptions

CONFIDENCE SCORING:
- 0.9-1.0: Crystal clear image with all details easily readable
- 0.7-0.8: Good quality with most details visible
- 0.5-0.6: Moderate quality with some details unclear
- 0.3-0.4: Poor quality but basic identification possible
- 0.0-0.2: Very poor quality, mostly guessing

IMAGE QUALITY ASSESSMENT:
- "clear": Sharp, well-lit, all text readable
- "blurry": Some blur but main features visible
- "dark": Poor lighting but details discernible
- "poor": Very difficult to make out details"""

        if text:
            prompt_text += f"\n\nTEXT CONTEXT:\n{text}"
        
        prompt_text += "\n\nExtract the product details as JSON following the ProductDetails schema exactly."
        
        return {
            "prompt_text": prompt_text,
            "image_path": image_path,
            "response_format": ProductDetails,
            "result_key": "product_details",
        }

class ProductEvaluationPrompt(BasePromptBuilder):
    """Prompt builder for evaluating product extraction quality."""
    
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Build prompt for evaluating product extraction results.
        
        Args:
            item: Dict containing:
                - image_path: str - Path to original image
                - extraction_result: Dict - The extracted product details
                - expected_details: Optional[Dict] - Known correct details for comparison
        """
        image_path = item["image_path"]
        extraction_result = item["extraction_result"]
        expected_details = item.get("expected_details")
        
        prompt_text = """Evaluate the quality of this product extraction by analyzing the original image and comparing it with the extracted details.

EVALUATION CRITERIA:

1. **Correct Identification**: Is the product type/category correctly identified?
2. **Accurate Details**: Are the extracted details (name, brand, price, etc.) accurate based on what's visible?
3. **Complete Extraction**: Were all clearly visible details captured?
4. **Schema Compliance**: Does the output properly follow the ProductDetails schema?
5. **Reasonable Confidence**: Is the confidence score appropriate for the image quality?

ASSESSMENT GUIDELINES:
- **Excellent**: All visible details correctly extracted, appropriate confidence, complete analysis
- **Good**: Most details correct, minor omissions or inaccuracies
- **Fair**: Basic identification correct but missing significant details
- **Poor**: Major errors in identification or significant missing information

WHAT TO CHECK:
- Product name matches what's visible on packaging/labels
- Brand identification is correct
- Price information is accurately extracted
- Color, size, material descriptions match the image
- Key features listed are actually visible
- Confidence score reflects actual image quality
- Category classification is appropriate"""

        if expected_details:
            prompt_text += f"\n\nKNOWN CORRECT DETAILS (for reference):\n{expected_details}"
        
        prompt_text += f"""

EXTRACTED DETAILS TO EVALUATE:
{extraction_result}

IMAGE PATH: {image_path}

Provide your evaluation following the ProductEvaluation schema."""
        
        return {
            "prompt_text": prompt_text,
            "image_path": image_path,
            "response_format": ProductEvaluation,
            "result_key": "evaluation",
        }

class ProductComparisonPrompt(BasePromptBuilder):
    """Prompt builder for comparing multiple product extractions."""
    
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Build prompt for comparing product extraction results.
        
        Args:
            item: Dict containing:
                - extractions: List[Dict] - Multiple extraction results to compare
                - image_paths: List[str] - Corresponding image paths
        """
        extractions = item["extractions"]
        image_paths = item.get("image_paths", [])
        
        prompt_text = """Compare these product extraction results and identify the most accurate and complete one.

COMPARISON CRITERIA:
1. Accuracy of product identification
2. Completeness of extracted details
3. Appropriate confidence scoring
4. Consistency with visible information

EXTRACTION RESULTS TO COMPARE:"""
        
        for i, extraction in enumerate(extractions):
            image_info = f" (Image: {image_paths[i]})" if i < len(image_paths) else ""
            prompt_text += f"\n\nResult {i+1}{image_info}:\n{extraction}"
        
        prompt_text += """

Analyze each result and provide:
1. Which extraction is most accurate overall
2. What details each extraction got right or wrong
3. Suggestions for improving the extraction process

Respond with your analysis following the ProductEvaluation schema, focusing on the best extraction."""
        
        return {
            "prompt_text": prompt_text,
            "response_format": ProductEvaluation,
            "result_key": "comparison",
        }
