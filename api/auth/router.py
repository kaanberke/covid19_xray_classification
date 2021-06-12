import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from api.auth import schema, crud
from api.utils import cryptoUtil
from fastapi.security import OAuth2PasswordRequestForm
from api.utils import jwtUtil, constantUtil, emailUtil

router = APIRouter(
    prefix="/api/v1"
)


@router.post("/auth/register", response_model=schema.UserShow)
async def register(user: schema.UserCreate):
    result = await crud.find_exist_user(user.email)

    if result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="User already registered.")

    user.password = cryptoUtil.hash_password(user.password)
    await crud.save_user(user)
    return {**user.dict()}


@router.post("/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    result = await crud.find_exist_user(form_data.username)

    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="User not found.")

    user = schema.UserCreate(**result)
    verified_password = cryptoUtil.verify_password(form_data.password, user.password)

    if not verified_password:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Invalid credentials.")

    access_token_expires = jwtUtil.timedelta(minutes=constantUtil.ACCESS_TOKEN_EXPIRE_MINUTE)
    access_token = await jwtUtil.create_access_token(
        data={"sub": form_data.username},
        expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_info": {
            "email": user.email,
            "fullname": user.fullname
        }
    }


@router.post("/auth/forgot-password")
async def forgot_password(request: schema.ForgotPassword):
    result = await crud.find_exist_user(request.email)

    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="User not found.")

    reset_code = str(uuid.uuid1())
    await crud.create_reset_code(request.email, reset_code)

    # Forgot password mail
    subject = "Forgot Password"
    recipients = [request.email]
    message = """
    
    <!DOCTYPE html>
    <html>
    <head>
        <title> Reset Password </title>
    </head>
    <body>
        <div style="width:100%;font-family: monospace;">
        <h1>Hello, {0:}</h1>
        <p>Looks like you've forgotten your password. You can change your password through the link.</p>
        <p>If you haven't requested for a password change, please kindly ignore this mail.</p>
        <a href="http://127.0.0.1/user/forgot-password?reset_password_token={1:}" style="box-sizing:border-box; border-color:#1f8feb; text-decoration:none; background-color:#1f8feb; border:solid 1px #1f8feb; border-radius:4px; color:#ffffff; font-size:16px; font-weight:bold; margin:0; padding:12px 24px; text-transformation:capitalize; display:inline-block" target="_blank">Reset Password</a>
    </body>
    </html>
    """.format(result.get("fullname"), reset_code)

    await emailUtil.send_mail(subject, recipients, message)
    return {
        "code": status.HTTP_200_OK,
        "message": "A password reset link has been sent.",
        "reset_code": reset_code,
    }


@router.patch("/auth/reset-password")
async def reset_password(request: schema.ResetPassword):
    reset_password_token = await crud.check_reset_password_token(request.reset_password_token)

    if not reset_password_token:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Reset password token has expired, please request a new one.")

    forgot_password_obj = schema.ForgotPassword(**reset_password_token)
    new_hashed_password = cryptoUtil.hash_password(request.new_password)
    await crud.reset_password(new_hashed_password, forgot_password_obj.email)

    await crud.disable_reset_code(request.reset_password_token, forgot_password_obj.email)

    return {
        "code": status.HTTP_200_OK,
        "message": "Password has been changed, successfully."
    }

