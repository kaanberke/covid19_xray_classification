from pathlib import Path
from time import time
from fastapi import APIRouter, Depends, UploadFile, File, status
import numpy as np
import cv2
from api.predict import predict, LABELS
from api.auth import schema as auth_schema
from api.images import crud as images_crud
from api.images import schema as images_schema
from api.utils import jwtUtil

router = APIRouter(
    prefix="/api/v1",
    tags=["Images"]
)


@router.post("/image/predict")
async def create_image_file(file: UploadFile = File(...),
                            current_user: auth_schema.UserShow = Depends(jwtUtil.get_current_user)):
    models_folder = Path("./api/models")
    start = time()

    try:
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        result = predict(models_folder, img)
        print(f"Took {time() - start:.2f} seconds..")

        await images_crud.save_image(contents, current_user.email, LABELS[result])
        id = await images_crud.get_last_image_id(current_user.email)
    except Exception as e:
        print(e)
        raise status.HTTP_422_UNPROCESSABLE_ENTITY

    return {
        "id": id,
        "filename": file.filename,
        "size": img.shape,
        "result": LABELS[result]
    }


@router.get("/image/result")
async def get_result_by_id(id: int, current_user: auth_schema.UserShow = Depends(jwtUtil.get_current_user)):
    result = await images_crud.get_result_by_id(id, current_user.email)
    return result
