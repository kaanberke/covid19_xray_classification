from sqlalchemy import Table, Column, Integer, String, DateTime, MetaData, Sequence, LargeBinary

metadata = MetaData()

users = Table(
    "users", metadata,
    Column("id", Integer, Sequence("user_id_seq"), primary_key=True),
    Column("email", String(100)),
    Column("password", String(100)),
    Column("fullname", String(50)),
    Column("created_on", DateTime),
    Column("status", String(1))
)

codes = Table(
    "codes", metadata,
    Column("id", Integer, Sequence("code_id_seq"), primary_key=True),
    Column("email", String(100)),
    Column("reset_code", String(50)),
    Column("status", String(1)),
    Column("expired_in", DateTime)
)

blacklists = Table(
    "blacklists", metadata,
    Column("token", String(250), unique=True),
    Column("email", String(100)),
)

images = Table(
    "images", metadata,
    Column("id", Integer, Sequence("image_id_seq"), primary_key=True),
    Column("image", LargeBinary),
    Column("result", String(50)),
    Column("created_on", DateTime),
    Column("uploaded_by", String(100))
)
