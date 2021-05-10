from time import time
from fastapi import APIRouter, Depends, UploadFile, File, status
import numpy as np
import cv2
from api.predict import predict, LABELS
from api.auth import schema as auth_schema
from api.images import crud as images_crud
from api.utils import jwtUtil

router = APIRouter(
    prefix="/api/v1",
    tags=["Images"]
)


@router.post("/image/predict")
async def create_image_file(file: UploadFile = File(...), current_user: auth_schema.UserShow = Depends(jwtUtil.get_current_user)):
    start = time()
    if "image" not in file.content_type:
        raise status.HTTP_422_UNPROCESSABLE_ENTITY

    contents = await file.read()
    await images_crud.save_image(contents, current_user.email)

    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, (224, 224, 1))
    result = predict("api/models/", img)
    print(f"Took {time()-start:.2f} seconds..")
    return {
        "filename": file.filename,
        "size":     img.shape,
        "result":   LABELS[result]
    }
