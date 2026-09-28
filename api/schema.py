from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator

class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=-18, le=-8)
    longitude: float = Field(..., ge=21, le=34)
    crop: Literal["Maize","Groundnuts"]
    province: str
    region: Literal["Rural","Urban"]
    planting_month: str
    farm_size_ha: float = Field(..., gt=0)
    soil_ph: Optional[float] = Field(None, ge=0, le=14)
    fertilizer_type: Optional[str] = None
    fertilizer_amount_kg_ha: Optional[float] = Field(None, ge=0)
    soil_type: Optional[str] = None
    crop_variety: Optional[str] = None
    previous_crop: Optional[str] = None
    water_source: Optional[str] = None
    district: Optional[str] = None
    ward: Optional[str] = None

    @field_validator("province")
    @classmethod
    def valid_province(cls, v):
        allowed = {"Central","Copperbelt","Eastern","Luapula","Lusaka","Muchinga",
                   "Northern","North Western","Southern","Western"}
        if v not in allowed:
            raise ValueError("Invalid province")
        return v

class PredictionResponse(BaseModel):
    predicted_yield_mt_per_ha: float
    predicted_total_harvest_mt: float
    yield_band: str
    risk_level: str
    risks: list[dict]
    recommendations: list[dict]
    model_features_missing: list[str]
    context: dict
