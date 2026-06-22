from app.features.users.model import User
from fastapi import Depends, APIRouter, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.features.users.schema import UserCreateReq, UserResponse, TokenResponse, UserLoginReq
from app.features.users.service import UserService
from app.core.deps import get_db, get_current_user
from app.core.schemas import APIResponse  

router = APIRouter(tags=["Authentication"])


@router.post("/register", response_model=APIResponse[UserResponse], status_code=status.HTTP_201_CREATED)
async def register(user_req: UserCreateReq, db: AsyncSession = Depends(get_db)):
    """Register a new user."""
    service = UserService(db)
    user =  await service.register_user(user_req)

    return APIResponse(
        message="User registered successfully",
        data=user,
    )


@router.post("/login", response_model=APIResponse[TokenResponse])
async def login(user_req: UserLoginReq, db: AsyncSession = Depends(get_db)):
    """Authenticate user and return access token."""
    service = UserService(db)
    token_data = await service.authenticate_user(user_req)
    
    return APIResponse(
        message="Login successful",
        data=token_data,
    )
    

@router.get("/me", response_model=APIResponse[UserResponse])
async def get_me(current_user: User = Depends(get_current_user)):
    """Get the current authenticated user's profile."""
    return APIResponse(
        message="User profile retrieved successfully",
        data=current_user,
    )