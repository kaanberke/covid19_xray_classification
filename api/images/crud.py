from api.utils.dbUtil import database
from starlette.responses import JSONResponse

def save_image(image: bytes, email: str, result: str):
    query = "INSERT INTO images VALUES (nextval('image_id_seq'), :image, :result, now() at time zone 'UTC', :email)"
    return database.execute(query, values={
        "image": image,
        "email": email,
        "result": result,
    })


def get_last_image_id(email: str):
    query = "SELECT id FROM images WHERE uploaded_by=:email ORDER BY created_on DESC LIMIT 1;"
    return database.execute(query, values={
        "email": email
    })


def get_result_by_id(id: int, email: str):
    query = "SELECT result, created_on FROM images WHERE uploaded_by=:email AND id=:id;"
    return database.fetch_one(query, values={
        "id": id,
        "email": email
    })
