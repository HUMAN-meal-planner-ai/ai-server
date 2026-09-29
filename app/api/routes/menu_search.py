from fastapi import APIRouter, Header
from pydantic import BaseModel

from app.agent.graph import run_menu_agent

router = APIRouter(prefix="/api/menus", tags=["메뉴 검색"])


class SearchRequest(BaseModel):
    query: str


@router.post("/search")
def search_menu(
    request: SearchRequest,
    authorization: str | None = Header(default=None),
):
    result = run_menu_agent(
        request.query,
        authorization=authorization,
    )

    return {
        "menus": result.get("menus", [])
    }