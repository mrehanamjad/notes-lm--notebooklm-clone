from typing import Generic, TypeVar, Optional, Any
from pydantic import BaseModel

T = TypeVar("T")

class APIResponse(BaseModel, Generic[T]):
    success: bool = True
    message: str = "Success"
    data: Optional[T] = None

class APIErrorResponse(BaseModel):
    success: bool = False
    error_type: str
    message: str
    details: Optional[Any] = None