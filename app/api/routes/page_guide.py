from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.rag.query_parser import extract_json, get_client

router = APIRouter(
    prefix="/api/guide",
    tags=["페이지 안내"],
)

# 안내 가능한 페이지 (roles가 없으면 로그인한 모든 사용자)
PAGES = [
    {
        "path": "/menus",
        "name": "메뉴 목록",
        "description": "등록된 메뉴 카탈로그를 검색하고 원가·영양 정보를 확인하는 화면. 날짜나 주간에 제공될 식단을 조회하는 곳은 아님",
    },
    {
        "path": "/menus/chat",
        "name": "메뉴 챗",
        "description": "재료, 분류, 가격 조건을 말하면 AI가 메뉴를 추천해 주는 화면",
    },
    {
        "path": "/budget",
        "name": "예산 분석",
        "description": "월간 예산, 식재료비 지출, 1인 목표 원가 대비 현황을 분석하는 화면",
    },
    {
        "path": "/prices",
        "name": "가격 예측",
        "description": "식재료 가격이 앞으로 오를지 내릴지 예측 결과를 보는 화면",
    },
    {
        "path": "/meal-plans",
        "name": "식단",
        "description": "일별 식단을 작성하고 관리하는 화면",
    },
    {
        "path": "/meal-plans/weekly",
        "name": "주간 식단",
        "description": "이번 주나 특정 주에 제공되거나 제공 예정인 메뉴·식단을 주간별로 조회하고 구성하는 화면",
    },
    {
        "path": "/my-page",
        "name": "마이페이지",
        "description": "내 계정과 시설 정보를 확인하고 수정하는 화면",
    },
    {
        "path": "/setup",
        "name": "시설 설정",
        "description": "시설명, 식수 인원, 목표 식재료비, 월간 예산을 설정하는 화면",
    },
    {
        "path": "/admin",
        "name": "관리자",
        "description": "전체 사용자와 시설을 관리하는 관리자 전용 화면",
        "roles": ["ADMIN"],
    },
    {
        "path": "/manager",
        "name": "시설 관리자",
        "description": "소속 시설의 운영 현황을 관리하는 시설 관리자 전용 화면",
        "roles": ["MANAGER"],
    },
]


class GuideRequest(BaseModel):
    question: str
    role: str | None = None


def allowed_pages(role: str | None) -> list[dict]:
    return [
        page
        for page in PAGES
        if "roles" not in page or role in page["roles"]
    ]


@router.post("")
async def guide_page(request: GuideRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해 주세요.")

    pages = allowed_pages(request.role)
    page_lines = "\n".join(
        f'- {page["path"]} ({page["name"]}): {page["description"]}'
        for page in pages
    )

    system_prompt = (
        "너는 MEALFIT 급식 관리 서비스의 페이지 안내 도우미다. "
        "사용자의 질문에 가장 알맞은 페이지를 아래 목록에서 고른다.\n"
        f"{page_lines}\n\n"
        "규칙:\n"
        "- 목록에 있는 path만 사용한다. 알맞은 페이지가 없으면 path는 null.\n"
        "- 사용자가 이번 주/다음 주/특정 주에 나오는 메뉴, 제공 예정 식단, 주간 메뉴를 조회하려는 경우 반드시 /meal-plans/weekly를 선택한다. 이 요청은 메뉴 카탈로그 조회(/menus)가 아니다.\n"
        "- /menus는 특정 주나 날짜와 무관하게 등록된 메뉴 자체를 검색하려는 경우에만 선택한다.\n"
        "- 메뉴 추천 요청은 /menus/chat으로 안내한다.\n"
        "- answer는 한두 문장의 친절한 한국어 안내문이다.\n"
        '- JSON만 반환한다: {"path": string|null, "answer": string}'
    )

    try:
        response = get_client().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )
        parsed = extract_json(response.choices[0].message.content or "")
    except Exception as exception:
        print("[ERROR]", repr(exception))
        raise HTTPException(
            status_code=503,
            detail="안내 서비스를 사용할 수 없습니다.",
        ) from exception

    path = parsed.get("path")
    matched = next((page for page in pages if page["path"] == path), None)
    answer = parsed.get("answer")

    if not isinstance(answer, str) or not answer.strip():
        answer = (
            f"{matched['name']} 페이지를 안내해 드릴게요."
            if matched
            else "해당 기능을 제공하는 페이지를 찾지 못했어요."
        )

    return {
        "answer": answer,
        "page": (
            {"path": matched["path"], "name": matched["name"]}
            if matched
            else None
        ),
    }
