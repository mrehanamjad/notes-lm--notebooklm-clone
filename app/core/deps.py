import uuid
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from jose import JWTError
from app.core.security import oauth2_scheme, decode_access_token
from app.database.session import AsyncSessionLocal
from app.features.users.model import User
from app.core.exceptions import AuthenticationException
from app.core.logger import logger


async def get_db():
    """Dependency to get an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def get_user_id_from_token(token: str) -> uuid.UUID:
    """Extract user ID from JWT token."""
    payload = decode_access_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise AuthenticationException("Invalid token: missing user ID")
    try:
        return uuid.UUID(user_id)
    except ValueError:
        raise AuthenticationException("Invalid user ID in token")
    except JWTError as e:
        logger.error(f"JWT decoding error: {e}")
        raise AuthenticationException("Invalid or expired token")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Dependency to get the current authenticated user."""
    token = credentials.credentials  # Extract raw JWT string from HTTPBearer
    user_id = get_user_id_from_token(token)
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise AuthenticationException("User not found")
    logger.info(f"Authenticated user: {user.username} (ID: {user.id})")
    return user
