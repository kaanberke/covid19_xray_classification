from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class UserShow(BaseModel):
    id: int = None
    email: str = Field(..., example="johndoe@gmail.com")
    fullname: str = Field(..., example="John DOE")
    created_on: Optional[datetime] = None
    status: str = None


class UserCreate(UserShow):
    password: str = Field(..., example="password")


class ForgotPassword(BaseModel):
    email: str


class ResetPassword(BaseModel):
    reset_password_token: str
    new_password: str