from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class ProductCategory(str, Enum):
    """Product category enumeration."""
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    FOOD = "food"
    BOOKS = "books"
    HOME = "home"
    SPORTS = "sports"
    BEAUTY = "beauty"
    AUTOMOTIVE = "automotive"
    OTHER = "other"

class ProductCondition(str, Enum):
    """Product condition enumeration."""
    NEW = "new"
    USED = "used"
    REFURBISHED = "refurbished"
    DAMAGED = "damaged"

class ProductDetails(BaseModel):
    """Comprehensive product details extracted from images and text."""
    
    # Basic product information
    name: str = Field(description="Product name or title")
    brand: Optional[str] = Field(default=None, description="Brand name if visible")
    category: ProductCategory = Field(description="Product category")
    
    # Pricing and availability
    price: Optional[float] = Field(default=None, description="Price if visible (numeric value only)")
    currency: Optional[str] = Field(default=None, description="Currency symbol or code")
    original_price: Optional[float] = Field(default=None, description="Original price if on sale")
    discount_percentage: Optional[float] = Field(default=None, description="Discount percentage if applicable")
    
    # Product characteristics
    color: Optional[str] = Field(default=None, description="Primary color of the product")
    size: Optional[str] = Field(default=None, description="Size information if available")
    material: Optional[str] = Field(default=None, description="Material composition if visible")
    condition: Optional[ProductCondition] = Field(default=None, description="Product condition")
    
    # Features and specifications
    key_features: List[str] = Field(default_factory=list, description="List of key features or specifications")
    description: Optional[str] = Field(default=None, description="Product description if available")
    
    # Visual characteristics
    packaging_type: Optional[str] = Field(default=None, description="Type of packaging (box, bag, bottle, etc.)")
    text_visible: bool = Field(default=False, description="Whether text is clearly visible on the product")
    logo_visible: bool = Field(default=False, description="Whether brand logo is visible")
    
    # Quality assessment
    image_quality: str = Field(description="Assessment of image quality: clear, blurry, dark, or poor")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in extraction accuracy (0-1)")

class ProductEvaluation(BaseModel):
    """Evaluation of product extraction quality."""
    
    # Accuracy metrics
    correct_identification: bool = Field(description="Is the product correctly identified?")
    accurate_details: bool = Field(description="Are the extracted details accurate?")
    complete_extraction: bool = Field(description="Are all visible details captured?")
    
    # Quality metrics
    schema_compliance: bool = Field(description="Does the output follow the schema correctly?")
    reasonable_confidence: bool = Field(description="Is the confidence score reasonable?")
    
    # Overall assessment
    overall_quality: str = Field(description="Overall quality: excellent, good, fair, or poor")
    missing_details: List[str] = Field(default_factory=list, description="List of details that should have been extracted but weren't")
    incorrect_details: List[str] = Field(default_factory=list, description="List of incorrectly extracted details")
    
    # Reasoning
    reasoning: Optional[str] = Field(default=None, description="Explanation of the evaluation")
    suggestions: Optional[str] = Field(default=None, description="Suggestions for improvement")
