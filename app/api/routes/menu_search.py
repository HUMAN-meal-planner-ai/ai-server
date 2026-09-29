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
        # 백엔드 인증·연결처럼 사용자가 조치할 수 있는 오류는 구체적인
        # 메시지와 함께 전달하여 프론트가 단순 연결 실패로 표시하지 않게 합니다.
        raise HTTPException(status_code=503, detail=str(exception)) from exception
    except Exception as exception:
        # 처리되지 않은 예외가 CORS 미들웨어 밖으로 전파되면 브라우저에는
        # 실제 원인 대신 "Failed to fetch"만 보일 수 있으므로 JSON 오류로 변환합니다.
        raise HTTPException(
            status_code=500,
            detail="AI 메뉴 추천 처리 중 오류가 발생했습니다.",
        ) from exception

    return {
        "menus": result.get("menus", [])
    }
