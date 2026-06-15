from datetime import datetime, timedelta, timezone
import uuid
from typing import Any, Optional, Union

from fastapi.security import HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from passlib.exc import UnknownHashError

from app.core.config import settings
from app.core.exceptions import AuthenticationException


# Password Hashing
pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto"
)

oauth2_scheme = HTTPBearer()


def hash_password(password: str) -> str:
    """Hash a plain-text password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against stored hash."""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except UnknownHashError:
        raise AuthenticationException("Invalid password hash format")


def create_access_token(
    user_id: Union[uuid.UUID, str],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create JWT access token."""
    expire = datetime.now(timezone.utc) + (
        expires_delta
        or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    payload = {
        "sub": str(user_id),
        "type": "access",
        "iat": datetime.now(timezone.utc),
        "exp": expire,
    }

    return jwt.encode(
        payload,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate JWT token."""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )

        if payload.get("type") != "access":
            raise AuthenticationException("Invalid token type")

        return payload

    except JWTError:
        raise AuthenticationException("Invalid or expired token")