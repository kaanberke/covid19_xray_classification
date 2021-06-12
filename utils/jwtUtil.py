import jwt
from jwt import PyJWTError
from pydantic import ValidationError
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from starlette import status

from api.auth import crud, schema
from api.utils import constantUtil


async def create_access_token(*, data:dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=constantUtil.ACCESS_TOKEN_EXPIRE_MINUTE))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, constantUtil.SECRET_KEY, algorithm=constantUtil.SIGNING_ALGORITHM)

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login"
)


def get_token_user(token: str = Depends(oauth2_scheme)):
    return token


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials.",
        headers={
            "WWW-Authenticate": "Bearer"
        }
    )

    try:
        payload = jwt.decode(token, constantUtil.SECRET_KEY, algorithms=[constantUtil.SIGNING_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception

        black_list_token = await crud.find_black_list_token(token)
        if black_list_token:
            raise credentials_exception

        result = await crud.find_exist_user(username)

        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail="User not found.")

        return schema.UserShow(**result)

    except (PyJWTError, ValidationError):
        raise credentials_exception


def get_current_active_user(current_user: schema.UserShow = Depends(get_current_user)):
    if current_user.status != '1':
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user."
        )

    return current_user
