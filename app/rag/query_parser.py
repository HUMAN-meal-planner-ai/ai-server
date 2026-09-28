import json

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI()


def parse_query(question: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={
            "type": "json_object"
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "급식 메뉴 검색 질문에서 검색 조건만 추출해서 "
                    "반드시 JSON으로 답하세요.\n\n"

                    "사용할 키는 다음과 같습니다.\n"
                    "- max_price\n"
                    "- category\n"
                    "- price_scope\n"
                    "- slot\n"
                    "- include_ingredients\n"
                    "- exclude_ingredients\n\n"

                    "규칙:\n"
                    "1. max_price는 최대 금액(원)입니다. 없으면 null.\n"
                    "2. category는 질문에 직접 언급된 음식 분류입니다. "
                    "예: 볶음류, 국류, 찌개류, 밥류. 없으면 null.\n"
                    "3. price_scope는 메뉴 하나의 가격이면 'menu', "
                    "식단 전체 가격이면 'meal', 불분명하거나 없으면 null.\n"
                    "4. slot은 식단 역할을 의미합니다.\n"
                    "   밥/주식 -> RICE\n"
                    "   국/탕/찌개 -> SOUP\n"
                    "   주찬 -> MAIN\n"
                    "   부찬/반찬 -> SIDE\n"
                    "   김치 -> KIMCHI\n"
                    "   그 외 명확한 경우 -> OTHER\n"
                    "   식단 역할이 없으면 null.\n"
                    "5. include_ingredients는 반드시 포함해야 하는 식재료 목록입니다.\n"
                    "6. exclude_ingredients는 제외해야 하는 식재료 목록입니다.\n"
                    "7. 식재료 조건이 없으면 빈 배열 []을 사용하세요.\n"
                    "8. 질문에 없는 조건은 절대 추측하지 마세요.\n"
                    "9. 메뉴 이름을 식재료라고 추측하지 마세요.\n\n"

                    "예시:\n"
                    "'양파 들어간 주찬 추천해줘'\n"
                    "-> include_ingredients=['양파'], slot='MAIN'\n\n"

                    "'돼지고기 빼고 국 추천해줘'\n"
                    "-> exclude_ingredients=['돼지고기'], slot='SOUP'\n\n"

                    "'두부랑 버섯 들어간 반찬 찾아줘'\n"
                    "-> include_ingredients=['두부', '버섯'], slot='SIDE'"
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    data = json.loads(
        response.choices[0].message.content
    )

    return {
        "max_price": data.get("max_price"),
        "category": data.get("category"),
        "price_scope": data.get("price_scope"),
        "slot": data.get("slot"),
        "include_ingredients": (
            data.get("include_ingredients") or []
        ),
        "exclude_ingredients": (
            data.get("exclude_ingredients") or []
        ),
    }