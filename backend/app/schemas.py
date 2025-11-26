from pydantic import BaseModel, Field, field_validator
from typing import List, Literal

class TransactionInput(BaseModel):

    features: List[float] = Field(
        ..., 
        min_items=30, 
        max_items=30, 
        description="[Time, V1, V2, ..., V28, Amount] are 30 numerical values respectively."
    )

    @field_validator('features')
    def check_features_length(cls, v):
        if len(v) != 30:
            raise ValueError('The model expects exactly 30 features.')
        return v

class PredictionOutput(BaseModel):

    is_fraud: bool
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Probability between 0 and 1")
    risk_level: Literal["CRITICAL", "HIGH", "MODERATE", "LOW"]