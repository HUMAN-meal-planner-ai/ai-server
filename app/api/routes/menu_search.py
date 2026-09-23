from fastapi import APIRouter
from pydantic import BaseModel

from app.rag.search_menus import search_menus

router = APIRouter(prefix="/api/menus", tags=["메뉴 검색"])


class SearchRequest(BaseModel):
    query: str


@router.post("/search")
def search_menu(request: SearchRequest):
    return {"menus": search_menus(request.query)}