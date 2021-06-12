from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.utils.dbUtil import database
from api.auth import router as auth_router
from api.users import router as users_router
from api.images import router as images_router


app = FastAPI(
    docs_urls="/docs",
    redoc_url="/redocs",
    title="Covid19 Finder",
    description="Covid19 Finder API <br>"
                "Fast and easy to use.",
    version="1.0",
    openapi_url="/openapi.json"
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await database.connect()


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


app.include_router(auth_router.router, tags=["Auth"])
app.include_router(users_router.router, tags=["Users"])
app.include_router(images_router.router, tags=["Images"])
