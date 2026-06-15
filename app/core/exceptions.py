from pydantic import BaseModel
from fastapi import HTTPException, status

class ErrorResponse(BaseModel):
    error_type: str
    message: str
    details: dict | None = None

class AppException(HTTPException):
    def __init__(self, status_code: int, error_type: str, message: str, details: dict | None = None):
        self.error_response = ErrorResponse(
                error_type=error_type, 
                message=message, 
                details=details
            )
        
        super().__init__(status_code=status_code, detail=self.error_response.dict())

class ValidationError(AppException):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_type="ValidationError",
            message=message,
            details=details
        )

class AuthenticationException(AppException):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_type="AuthenticationError",
            message=message,
            details=details
        )

class AuthorizationException(AppException):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            error_type="AuthorizationError",
            message=message,
            details=details
        )

class NotFoundException(AppException):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_type="NotFoundError",
            message=message,
            details=details
        )

class ConflictException(AppException):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            error_type="ConflictError",
            message=message,
            details=details
        )

class InternalServerError(AppException):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_type="InternalServerError",
            message=message,
            details=details
        )