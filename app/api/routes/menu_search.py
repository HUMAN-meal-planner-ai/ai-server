from fastapi import APIRouter, Header, HTTPException
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
    try:
        result = run_menu_agent(
            request.query,
            authorization=authorization,
        )
    except RuntimeError as exception:
        raise HTTPException(status_code=503, detail=str(exception)) from exception

    except Exception as exception:
        print("[ERROR]", repr(exception))
        raise HTTPException(
            status_code=500,
            detail=str(exception),
        ) from exception

    return {
        "menus": result.get("menus", [])
    }