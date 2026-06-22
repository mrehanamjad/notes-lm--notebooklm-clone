from fastapi import status

class AppException(Exception):
    """Base exception for all custom application errors."""
    def __init__(self, status_code: int, error_type: str, message: str, details: dict | None = None):
        self.status_code = status_code
        self.error_type = error_type
        self.message = message
        self.details = details
        super().__init__(message)

class ValidationError(AppException):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_type="VALIDATION_ERROR",
            message=message,
            details=details
        )

class AuthenticationException(AppException):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_type="AUTHENTICATION_ERROR",
            message=message,
            details=details
        )

class AuthorizationException(AppException):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            error_type="AUTHORIZATION_ERROR",
            message=message,
            details=details
        )

class NotFoundException(AppException):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_type="NOT_FOUND_ERROR",
            message=message,
            details=details
        )

class ConflictException(AppException):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            error_type="CONFLICT_ERROR",
            message=message,
            details=details
        )

class InternalServerError(AppException):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_type="INTERNAL_SERVER_ERROR",
            message=message,
            details=details
        )


class BadRequestException(AppException):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_type="BAD_REQUEST_ERROR",
            message=message,
            details=details
        )

