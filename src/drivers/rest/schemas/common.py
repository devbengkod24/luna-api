from pydantic import BaseModel

class ErrorResponse(BaseModel):
    """Response model for error."""
    error: str
    detail: str