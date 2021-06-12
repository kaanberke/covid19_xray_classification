from typing import List
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from starlette.config import Config

config = Config(".env")
conf = ConnectionConfig(
    MAIL_USERNAME=config("MAIL_USERNAME"),
    MAIL_PASSWORD=config("MAIL_PASSWORD"),
    MAIL_FROM=config("MAIL_FROM"),
    MAIL_PORT=config("MAIL_PORT"),
    MAIL_SERVER=config("MAIL_SERVER"),
    MAIL_TLS=config("MAIL_TLS"),
    MAIL_SSL=config("MAIL_SSL"),
    USE_CREDENTIALS=config("USE_CREDENTIALS")
)


async def send_mail(subject: str, recipients: List, message: str):
    message = MessageSchema(
        subject=subject,
        recipients=recipients,
        body=message,
        subtype="html",
    )
    fm = FastMail(conf)
    await fm.send_message(message)
