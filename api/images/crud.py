from api.utils.dbUtil import database


def save_image(image: bytes, email: str):
    query = "INSERT INTO images VALUES (nextval('image_id_seq'), :image, now() at time zone 'UTC', :email)"
    return database.execute(query, values={
        "image": image,
        "email": email,
    })
