from pydantic import BaseModel, Field


class Intent(BaseModel):
    action: str
    confidence: float = Field(ge=0.0, le=1.0)
    brief: str = ""
