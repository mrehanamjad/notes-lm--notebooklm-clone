from gunicorn.config import User
from fastapi import Depends, APIRouter, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.features.users.schema import UserCreateReq, UserResponse, TokenResponse, UserLoginReq
from app.features.users.service import UserService
from app.core.deps import get_db, get_current_user

router = APIRouter(tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_req: UserCreateReq, db: AsyncSession = Depends(get_db)):
    """Register a new user."""
    service = UserService(db)
    return await service.register_user(user_req)


@router.post("/login", response_model=TokenResponse)
async def login(user_req: UserLoginReq, db: AsyncSession = Depends(get_db)):
    """Authenticate user and return access token."""
    service = UserService(db)
    return await service.authenticate_user(user_req)
    

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get the current authenticated user's profile."""
    return current_user