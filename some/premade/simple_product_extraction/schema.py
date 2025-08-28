"""
Product Data Schema

Defines the structured output format for product information extraction.
Uses Pydantic for validation and type safety.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class ProductCategory(str, Enum):
    """Common product categories for classification."""
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    HOME_GARDEN = "home_garden"
    SPORTS = "sports"
    BOOKS = "books"
    HEALTH_BEAUTY = "health_beauty"
    AUTOMOTIVE = "automotive"
    TOYS = "toys"
    FOOD_BEVERAGE = "food_beverage"
    OTHER = "other"

class ProductData(BaseModel):
    """
    Structured product information extracted from text descriptions.
    
    This schema captures essential product details that are commonly
    found in product descriptions, reviews, or marketing materials.
    """
    name: str = Field(description="Product name or title")
    
    price: Optional[float] = Field(
        default=None, 
        description="Product price in USD, if mentioned"
    )
    
    category: Optional[ProductCategory] = Field(
        default=None,
        description="Product category classification"
    )
    
    brand: Optional[str] = Field(
        default=None,
        description="Brand or manufacturer name"
    )
    
    features: List[str] = Field(
        default_factory=list,
        description="List of product features, benefits, or specifications"
    )
    
    description: Optional[str] = Field(
        default=None,
        description="Brief product description or summary"
    )
    
    availability: Optional[str] = Field(
        default=None,
        description="Availability status (in stock, out of stock, etc.)"
    )
    
    rating: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=5.0,
        description="Product rating out of 5 stars, if mentioned"
    )
    
    model_number: Optional[str] = Field(
        default=None,
        description="Product model or SKU number"
    )
