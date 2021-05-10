from api.utils.dbUtil import database
from api.auth import schema


def find_exist_user(email: str):
    query = "SELECT * FROM users WHERE status='1' AND email=:email"
    return database.fetch_one(query, values={"email": email})


def save_user(user: schema.UserCreate):
    query = "INSERT INTO users VALUES (nextval('user_id_seq'), :email, :password, :fullname, now() at time zone 'UTC', '1')"
    return database.execute(query, values={
        "email": user.email,
        "password": user.password,
        "fullname": user.fullname
    })


def create_reset_code(email: str, reset_code: str):
    query = "INSERT INTO codes VALUES (nextval('code_id_seq'), :email, :reset_code, '1', now() at time zone 'UTC')"
    return database.execute(query, values={
        "email": email,
        "reset_code": reset_code
    })


def check_reset_password_token(reset_password_token: str):
    query = "SELECT * FROM codes WHERE status='1' AND reset_code=:reset_password_token AND expired_in >= now() at time zone 'UTC' - interval '10 minutes'"
    return database.fetch_one(query, values={"reset_password_token": reset_password_token})


def reset_password(new_hashed_password: str, email: str):
    query = "UPDATE users SET password=:password WHERE email=:email"
    return database.execute(query, values={"password": new_hashed_password, "email": email})


def disable_reset_code(reset_password_token: str, email: str):
    query = "UPDATE codes SET status='0' WHERE status='1' AND reset_code=:reset_code AND email=:email"
    return database.execute(query, values={"reset_code": reset_password_token, "email": email})
