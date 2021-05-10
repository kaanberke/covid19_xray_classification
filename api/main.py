from fastapi import FastAPI
from api.utils.dbUtil import database
from api.auth import router as auth_router
from api.users import router as users_router


app = FastAPI(
    docs_urls="/docs",
    redoc_url="/redocs",
    title="Covid19 Finder",
    description="Covid19 Finder API <br>"
                "Fast and easy to use.",
    version="1.0",
    openapi_url="/openapi.json"
)


@app.on_event("startup")
async def startup():
    await database.connect()


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


app.include_router(auth_router.router, tags=["Auth"])
app.include_router(users_router.router, tags=["Users"])
