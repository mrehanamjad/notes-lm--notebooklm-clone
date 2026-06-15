import re
from app.core.exceptions import ValidationError

class PasswordValidator:
    """Validator class for password strength."""

    @staticmethod
    def validate(password: str) -> None:
        """Validate the password against defined strength criteria."""
        if len(password) < 8:
            raise ValidationError("Password must be at least 8 characters long.")
        if not re.search(r"[A-Z]", password):
            raise ValidationError("Password must contain at least one uppercase letter.")
        if not re.search(r"[a-z]", password):
            raise ValidationError("Password must contain at least one lowercase letter.")
        if not re.search(r"\d", password):
            raise ValidationError("Password must contain at least one digit.")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            raise ValidationError("Password must contain at least one special character.")
        
class EmailValidator:
    """Validator class for email format."""

    @staticmethod
    def validate(email: str) -> None:
        """Validate the email format."""
        email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        if not re.match(email_regex, email):
            raise ValidationError("Invalid email format.")
        
class StringValidator:
    """Validator class for general string fields."""

    @staticmethod
    def validate_non_empty(value: str, field_name: str) -> None:
        """Validate that the string is not empty."""
        if not value or value.strip() == "":
            raise ValidationError(f"{field_name} cannot be empty.")
        
    @staticmethod
    def validate_length(value: str, field_name: str, min_length: int = 1, max_length: int = 255) -> None:
        """Validate that the string length is within specified bounds."""
        if len(value) < min_length:
            raise ValidationError(f"{field_name} must be at least {min_length} characters long.")
        if len(value) > max_length:
            raise ValidationError(f"{field_name} cannot be longer than {max_length} characters.")