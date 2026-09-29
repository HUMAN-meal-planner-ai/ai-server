import json

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI()


def parse_query(question: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={
            "type": "json_object"
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "급식 메뉴 검색 질문에서 검색 조건만 추출해서 "
                    "반드시 JSON 객체로 답하세요.\n\n"

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
                    "예: 볶음류, 국류, 찌개류, 밥류. "
                    "없으면 null.\n"

                    "3. price_scope는 메뉴 하나의 가격이면 'menu', "
                    "식단 전체 가격이면 'meal', "
                    "가격 조건이 없거나 불분명하면 null.\n"

                    "4. slot은 식단 역할을 의미합니다.\n"
                    "   밥/주식 -> RICE\n"
                    "   국/탕/찌개 -> SOUP\n"
                    "   주찬 -> MAIN\n"
                    "   부찬/반찬 -> SIDE\n"
                    "   김치 -> KIMCHI\n"
                    "   그 외 명확한 경우 -> OTHER\n"
                    "   식단 역할이 없으면 null.\n"

                    "5. include_ingredients는 반드시 포함해야 하는 "
                    "식재료 목록입니다.\n"

                    "6. exclude_ingredients는 반드시 제외해야 하는 "
                    "식재료 목록입니다.\n"

                    "7. 식재료 조건이 없으면 빈 배열 []을 사용하세요.\n"

                    "8. 질문에 없는 조건은 절대 추측하지 마세요.\n"

                    "9. 메뉴 이름을 식재료라고 임의로 추측하지 마세요.\n"

                    "10. 식재료 이름과 함께 '추천', '찾아줘', "
                    "'메뉴' 등의 표현이 나오면 별도의 제외 표현이 없는 한 "
                    "해당 식재료를 include_ingredients에 넣으세요.\n"

                    "11. '빼고', '제외', '없이', '안 들어간', "
                    "'못 먹는다', '알레르기'처럼 명확한 제외 의미가 있을 때만 "
                    "exclude_ingredients에 넣으세요.\n"

                    "12. 포함 조건과 제외 조건을 절대로 반대로 해석하지 마세요.\n\n"

                    "예시 1:\n"
                    "질문: 양파 들어간 주찬 추천해줘\n"
                    "답:\n"
                    "{"
                    "\"max_price\": null, "
                    "\"category\": null, "
                    "\"price_scope\": null, "
                    "\"slot\": \"MAIN\", "
                    "\"include_ingredients\": [\"양파\"], "
                    "\"exclude_ingredients\": []"
                    "}\n\n"

                    "예시 2:\n"
                    "질문: 돼지고기 주찬 추천해줘\n"
                    "답:\n"
                    "{"
                    "\"max_price\": null, "
                    "\"category\": null, "
                    "\"price_scope\": null, "
                    "\"slot\": \"MAIN\", "
                    "\"include_ingredients\": [\"돼지고기\"], "
                    "\"exclude_ingredients\": []"
                    "}\n\n"

                    "예시 3:\n"
                    "질문: 3000원 이하 돼지고기 주찬 추천해줘\n"
                    "답:\n"
                    "{"
                    "\"max_price\": 3000, "
                    "\"category\": null, "
                    "\"price_scope\": \"menu\", "
                    "\"slot\": \"MAIN\", "
                    "\"include_ingredients\": [\"돼지고기\"], "
                    "\"exclude_ingredients\": []"
                    "}\n\n"

                    "예시 4:\n"
                    "질문: 돼지고기 빼고 국 추천해줘\n"
                    "답:\n"
                    "{"
                    "\"max_price\": null, "
                    "\"category\": null, "
                    "\"price_scope\": null, "
                    "\"slot\": \"SOUP\", "
                    "\"include_ingredients\": [], "
                    "\"exclude_ingredients\": [\"돼지고기\"]"
                    "}\n\n"

                    "예시 5:\n"
                    "질문: 두부랑 버섯 들어간 반찬 찾아줘\n"
                    "답:\n"
                    "{"
                    "\"max_price\": null, "
                    "\"category\": null, "
                    "\"price_scope\": null, "
                    "\"slot\": \"SIDE\", "
                    "\"include_ingredients\": [\"두부\", \"버섯\"], "
                    "\"exclude_ingredients\": []"
                    "}"
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

    include_ingredients = (
        data.get("include_ingredients") or []
    )

    exclude_ingredients = (
        data.get("exclude_ingredients") or []
    )

    exclude_keywords = [
        "빼",
        "뺀",
        "빼고",
        "빼줘",
        "제외",
        "제외한",
        "없이",
        "없는",
        "안 들어",
        "안들어",
        "못 먹",
        "못먹",
        "안 먹",
        "안먹",
        "알레르기",
    ]

    has_exclude_expression = any(
        keyword in question
        for keyword in exclude_keywords
    )

    # 제외 표현이 있는데 include로 잘못 들어간 경우
    if (
        has_exclude_expression
        and include_ingredients
        and not exclude_ingredients
    ):
        exclude_ingredients.extend(
            include_ingredients
        )
        include_ingredients = []

    # 제외 표현이 없는데 exclude로 잘못 들어간 경우
    elif (
        exclude_ingredients
        and not has_exclude_expression
    ):
        include_ingredients.extend(
            exclude_ingredients
        )
        exclude_ingredients = []

    include_ingredients = list(
        dict.fromkeys(include_ingredients)
    )

    exclude_ingredients = list(
        dict.fromkeys(exclude_ingredients)
    )

    return {
        "max_price": data.get("max_price"),
        "category": data.get("category"),
        "price_scope": data.get("price_scope"),
        "slot": data.get("slot"),
        "include_ingredients": include_ingredients,
        "exclude_ingredients": exclude_ingredients,
    }