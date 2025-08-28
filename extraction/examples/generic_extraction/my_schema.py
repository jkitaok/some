from __future__ import annotations

from pydantic import BaseModel
from typing import List, Optional

class ProductSpec(BaseModel):
    name: str
    price: float
    features: List[str]

class BasicEvaluation(BaseModel):
    correct: bool
    formatted: bool
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
