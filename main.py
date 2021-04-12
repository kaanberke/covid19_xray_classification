from fastapi import FastAPI, File, UploadFile, status
import uvicorn

import cv2
import numpy as np
from predict import predict, LABELS

from time import time

app = FastAPI()


@app.post("/image/", tags=["image"])
async def create_image_file(file: UploadFile = File(...)):
    start = time()
    if "image" not in file.content_type:
        raise status.HTTP_422_UNPROCESSABLE_ENTITY

    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, (224, 224, 1))
    result = predict("./models/", img)
    print(f"Took {time()-start:.2f} seconds..")
    return {
        "filename": file.filename,
        "size": img.shape,
        "result": LABELS[result]
    }


def run():
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"][
        "fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["access"][
        "fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    uvicorn.run("main:app",
                log_config=log_config,
                debug=True,
                reload=True,
                port=8000)


if __name__ == "__main__":
    run()
