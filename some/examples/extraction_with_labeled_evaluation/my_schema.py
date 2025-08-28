from __future__ import annotations

from pydantic import BaseModel
from typing import List

class ProductSpec(BaseModel):
    name: str
    price: float
    features: List[str]
