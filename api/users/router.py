from fastapi import APIRouter, Depends, HTTPException
from starlette import status

from api.auth import schema as auth_schema, crud as auth_crud
from api.users import schema as user_schema, crud as user_crud

from api.utils import jwtUtil, cryptoUtil

router = APIRouter(
    prefix="/api/v1"
)


@router.get("/user/profile")
async def get_user_profile(current_user: auth_schema.UserShow = Depends(jwtUtil.get_current_user)):
    return current_user


@router.patch("/user/profile")
async def update_profile(request: user_schema.UserUpdate, current_user: auth_schema.UserShow = Depends(jwtUtil.get_current_user)):
    await user_crud.update_user(request, current_user)
    return {
        "status_code": status.HTTP_200_OK,
        "detail": "User updated successfully."
    }


@router.delete("/user/profile")
async def deactivate_account(current_user: auth_schema.UserShow = Depends(jwtUtil.get_current_active_user)):
    await user_crud.deactivate_user(current_user)
    return {
        "status_code": status.HTTP_200_OK,
        "detail": "User account has been deactivated, successfully."
    }


@router.patch("/user/change-password")
async def change_password(
    change_password_obj: user_schema.ChangePassword,
    current_user: auth_schema.UserShow = Depends(jwtUtil.get_current_active_user)
):
    result = await auth_crud.find_exist_user(current_user.email)
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    user = auth_schema.UserCreate(**result)
    valid = cryptoUtil.verify_password(change_password_obj.current_password, user.password)
    if not valid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Current password is invalid.")

    change_password_obj.new_password = cryptoUtil.hash_password(change_password_obj.new_password)
    await user_crud.change_password(change_password_obj, current_user)
    return {
        "status_code": status.HTTP_200_OK,
        "detail": "Password has been changed, successfully."
    }


@router.get("/user/logout")
async def logout(
     token: str = Depends(jwtUtil.get_token_user),
     current_user: auth_schema.UserShow = Depends(jwtUtil.get_current_active_user)
):
    await user_crud.save_black_list_token(token, current_user)
    return {
        "status_code": status.HTTP_200_OK,
        "detail": "User logged out, successfully."
    }
