from api.users import schema as user_schema
from api.auth import schema as auth_schema
from api.utils.dbUtil import database


def update_user(request: user_schema.UserUpdate, current_user: auth_schema.UserShow):
    query = "UPDATE users SET fullname=:fullname WHERE email=:email"
    return database.execute(query, values={"fullname": request.fullname, "email": current_user.email})


def deactivate_user(current_user: auth_schema.UserShow):
    query = "UPDATE users SET status= '0' WHERE email=:email AND fullname=:fullname"
    return database.execute(query, values={"fullname": current_user.fullname, "email": current_user.email})


def change_password(change_password_obj: user_schema.ChangePassword, current_user: auth_schema.UserShow):
    query = "UPDATE users SET password=:password WHERE status='1' AND email=:email"
    return database.execute(query, values={"password": change_password_obj.new_password, "email": current_user.email})


def save_black_list_token(token: str, current_user: auth_schema.UserShow):
    query = "INSERT INTO blacklists VALUES(:token, :email)"
    return database.execute(query, values={"token": token, "email": current_user.email})