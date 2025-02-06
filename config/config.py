from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class Config(BaseModel):
    dest_folder: str
    sample_count: int
    COEFFS: Dict[str, Any] = Field(default_factory=dict)
    INIT: Optional[List[List[float]]] = None