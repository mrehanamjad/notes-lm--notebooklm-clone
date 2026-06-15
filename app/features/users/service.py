from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from app.features.users.repository import UserRepository
from app.features.users.model import User
from app.features.users.schema import UserCreateReq, UserLoginReq, TokenResponse
from app.core.security import hash_password, verify_password, create_access_token
from app.core.exceptions import ValidationError, AuthenticationException, ConflictException
from app.core.logger import logger
from app.core.validator import PasswordValidator, EmailValidator


class UserService:
    def __init__(self, db: AsyncSession):
        self.repository = UserRepository(db)

    async def register_user(self, user_req: UserCreateReq) -> User:
        """Register a new user."""
        # Validate inputs
        EmailValidator.validate(user_req.email)
        PasswordValidator.validate(user_req.password)

        # Check if user already exists
        existing_user = await self.repository.get_user_by_email(user_req.email)
        if existing_user:
            logger.error(f"Registration failed: User with this email already exists - {user_req.email}")
            raise ConflictException(message="User with this email already exists")

        # Create new user
        hashed_password = hash_password(user_req.password)
        new_user = User(
            username=user_req.username,
            email=user_req.email,
            password=hashed_password,
        )

        try:
            created_user = await self.repository.create_user(new_user)
            logger.info(f"User registered successfully: {created_user.username} (ID: {created_user.id})")
            return created_user
        except IntegrityError as e:
            logger.error(f"Database error during registration: {str(e)}")
            raise ValidationError(message="Failed to register user", details={"error": str(e)})

    async def authenticate_user(self, user_req: UserLoginReq) -> TokenResponse:
        """Authenticate user and return access token."""
        # Get user by email
        user = await self.repository.get_user_by_email(user_req.email)
        
        if not user:
            logger.error(f"Authentication failed: User not found - {user_req.email}")
            raise AuthenticationException(message="Invalid email or password")

        # Verify password
        if not verify_password(user_req.password, user.password):
            logger.error(f"Authentication failed: Incorrect password for user - {user_req.email}")
            raise AuthenticationException(message="Invalid email or password")

        # Create access token (user.id is UUID, create_access_token accepts UUID)
        access_token = create_access_token(user_id=user.id)

        logger.info(f"User authenticated successfully: {user.username} (ID: {user.id})")

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user_id=user.id,
            email=user.email
        )