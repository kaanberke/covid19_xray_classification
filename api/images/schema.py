from datetime import datetime
from typing import Optional
from fastapi import File
from pydantic import BaseModel


class ImageBase(BaseModel):
    id: int = None
    image: bytes = File(...)


class ImageShow(ImageBase):
    created_on: Optional[datetime] = None
    uploaded_by: str
