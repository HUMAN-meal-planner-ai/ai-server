from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.agent.graph import run_menu_agent

router = APIRouter(
    prefix="/api/menus",
    tags=["메뉴 검색"],
)


# 메뉴 검색 요청 데이터
class SearchRequest(BaseModel):
    query: str


# AI 메뉴 검색 API
@router.post("/search")
async def search_menu(
    request: SearchRequest,
    authorization: str | None = Header(default=None),
):
    try:
        # LangGraph 메뉴 추천 Agent 실행
        result = await run_menu_agent(
            request.query,
            authorization=authorization,
        )

    # 외부 서비스 또는 데이터 연결 오류
    except RuntimeError as exception:
        raise HTTPException(
            status_code=503,
            detail=str(exception),
        ) from exception

    # 그 외 서버 오류
    except Exception as exception:
        print("[ERROR]", repr(exception))

        raise HTTPException(
            status_code=500,
            detail=str(exception),
        ) from exception

    # 검색된 메뉴 목록 반환
    return {
        "menus": result.get("menus", [])
    }