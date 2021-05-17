from api.utils.dbUtil import database


def save_image(image: bytes, email: str, result: str):
    query = "INSERT INTO images VALUES (nextval('image_id_seq'), :image, :result, now() at time zone 'UTC', :email)"
    return database.execute(query, values={
        "image": image,
        "email": email,
        "result": result,
    })
