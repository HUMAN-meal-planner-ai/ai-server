import json
import re
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI


ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


@lru_cache(maxsize=1)
def get_client():
    env = dotenv_values(ENV_PATH)

    return OpenAI(
        api_key=env.get("OPENAI_API_KEY")
    )


def extract_json(text: str) -> dict:
    if not text:
        return {}

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"```$",
            "",
            text,
        )

        text = text.strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL,
        )

        if not match:
            return {}

        try:
            return json.loads(
                match.group()
            )

        except json.JSONDecodeError:
            return {}


def parse_number(value):
    if value is None:
        return None

    if isinstance(
        value,
        (int, float),
    ):
        return value

    text = str(value).strip()

    if not text:
        return None

    text = text.replace(",", "")

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        text,
    )

    if not match:
        return None

    number = float(
        match.group()
    )

    if number.is_integer():
        return int(number)

    return number


def normalize_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        return [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

    return []


def apply_numeric_regex(
    query: str,
    result: dict,
):
    price_patterns = [
        (
            r"(\d[\d,]*)\s*원\s*(?:이하|미만|안쪽|까지)",
            "max_price",
        ),
        (
            r"(\d[\d,]*)\s*원\s*(?:이상|초과)",
            "min_price",
        ),
    ]

    for pattern, key in price_patterns:
        match = re.search(
            pattern,
            query,
        )

        if match:
            result[key] = int(
                match.group(1).replace(
                    ",",
                    "",
                )
            )

    weight_patterns = [
        (
            r"(\d+(?:\.\d+)?)\s*(?:g|그램)\s*(?:이하|미만|까지)",
            "max_weight",
        ),
        (
            r"(\d+(?:\.\d+)?)\s*(?:g|그램)\s*(?:이상|초과)",
            "min_weight",
        ),
    ]

    for pattern, key in weight_patterns:
        match = re.search(
            pattern,
            query,
            re.IGNORECASE,
        )

        if match:
            result[key] = float(
                match.group(1)
            )

    calorie_patterns = [
        (
            r"(\d+(?:\.\d+)?)\s*(?:kcal|칼로리)\s*(?:이하|미만|까지)",
            "max_calories",
        ),
        (
            r"(\d+(?:\.\d+)?)\s*(?:kcal|칼로리)\s*(?:이상|초과)",
            "min_calories",
        ),
    ]

    for pattern, key in calorie_patterns:
        match = re.search(
            pattern,
            query,
            re.IGNORECASE,
        )

        if match:
            result[key] = float(
                match.group(1)
            )

    protein_patterns = [
        (
            r"단백질\s*(\d+(?:\.\d+)?)\s*g?\s*(?:이하|미만|까지)",
            "max_protein",
        ),
        (
            r"단백질\s*(\d+(?:\.\d+)?)\s*g?\s*(?:이상|초과)",
            "min_protein",
        ),
    ]

    for pattern, key in protein_patterns:
        match = re.search(
            pattern,
            query,
            re.IGNORECASE,
        )

        if match:
            result[key] = float(
                match.group(1)
            )

    sodium_patterns = [
        (
            r"나트륨\s*(\d+(?:\.\d+)?)\s*mg?\s*(?:이하|미만|까지)",
            "max_sodium",
        ),
        (
            r"나트륨\s*(\d+(?:\.\d+)?)\s*mg?\s*(?:이상|초과)",
            "min_sodium",
        ),
    ]

    for pattern, key in sodium_patterns:
        match = re.search(
            pattern,
            query,
            re.IGNORECASE,
        )

        if match:
            result[key] = float(
                match.group(1)
            )


def parse_query(query: str) -> dict:
    default_result = {
        "min_price": None,
        "max_price": None,
        "category": None,
        "menu_keyword": None,
        "price_scope": None,
        "slot": None,
        "include_ingredients": [],
        "exclude_ingredients": [],
        "min_weight": None,
        "max_weight": None,
        "min_calories": None,
        "max_calories": None,
        "min_protein": None,
        "max_protein": None,
        "min_sodium": None,
        "max_sodium": None,
    }

    if not query or not query.strip():
        return default_result

    prompt = f"""
너는 단체급식 메뉴 검색 조건을 추출하는 파서다.

사용자 문장에서 검색 조건을 추출해서
반드시 JSON 객체만 반환해라.

사용자 질문:
{query}

반환 형식:
{{
  "min_price": null,
  "max_price": null,
  "category": null,
  "menu_keyword": null,
  "price_scope": null,
  "slot": null,
  "include_ingredients": [],
  "exclude_ingredients": [],
  "min_weight": null,
  "max_weight": null,
  "min_calories": null,
  "max_calories": null,
  "min_protein": null,
  "max_protein": null,
  "min_sodium": null,
  "max_sodium": null
}}

규칙:

1. category
DB의 넓은 메뉴 분류를 의미한다.

예:
- 면류
- 밥류
- 국 및 탕류
- 찌개 및 전골류
- 볶음류
- 튀김류
- 조림류
- 구이류
- 김치류
- 떡류
- 과자 및 빵류

사용자가 "면류 추천", "밥류 추천"처럼
넓은 분류 자체를 말한 경우에만 category에 넣는다.

2. menu_keyword
사용자가 특정한 메뉴 종류나 메뉴명을 말하면
그 핵심 단어를 그대로 넣는다.

예:
"국수 추천해줘"
→ menu_keyword: "국수"

"라면 추천해줘"
→ menu_keyword: "라면"

"파스타 추천해줘"
→ menu_keyword: "파스타"

"볶음밥 추천해줘"
→ menu_keyword: "볶음밥"

"닭갈비 추천해줘"
→ menu_keyword: "닭갈비"

"카레 메뉴 추천해줘"
→ menu_keyword: "카레"

중요:
"면류", "밥류", "국류"처럼
DB의 넓은 분류를 말한 경우에는
menu_keyword를 넣지 않는다.

사용자가 말하지 않은 메뉴명을
임의로 만들어내지 않는다.

3. include_ingredients
반드시 들어가야 하는 식재료를 넣는다.

예:
"돼지고기 들어간 메뉴"
→ ["돼지고기"]

4. exclude_ingredients
들어가면 안 되는 식재료를 넣는다.

예:
"돼지고기 안 들어간 메뉴"
→ ["돼지고기"]

"돼지고기 없는 국수"
→ exclude_ingredients: ["돼지고기"]
→ menu_keyword: "국수"

5. 가격
"3000원 이하"
→ max_price: 3000

"1000원 이상"
→ min_price: 1000

메뉴 1개의 가격이면 price_scope는 "menu".
식단 전체 예산이면 price_scope는 "meal".

6. slot
밥, 국, 메인 반찬, 서브 반찬, 김치 등
식단 슬롯이 명확할 때만 사용한다.

가능한 값:
RICE
SOUP
MAIN
SIDE
KIMCHI
OTHER

7. 영양 조건
칼로리, 단백질, 나트륨, 중량 조건을
숫자로 추출한다.

8. 조건이 없으면 null 또는 빈 배열을 사용한다.

설명 문장은 쓰지 말고 JSON만 반환해라.
"""

    try:
        response = get_client().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content":
                        "사용자 자연어에서 메뉴 검색 조건을 추출한다. "
                        "반드시 JSON만 반환한다.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        content = (
            response.choices[0]
            .message.content
            or ""
        )

        parsed = extract_json(
            content
        )

    except Exception as e:
        print(
            "[QUERY PARSER ERROR]",
            e,
        )

        parsed = {}

    result = {
        **default_result,
        **parsed,
    }

    result["min_price"] = parse_number(
        result.get("min_price")
    )

    result["max_price"] = parse_number(
        result.get("max_price")
    )

    result["min_weight"] = parse_number(
        result.get("min_weight")
    )

    result["max_weight"] = parse_number(
        result.get("max_weight")
    )

    result["min_calories"] = parse_number(
        result.get("min_calories")
    )

    result["max_calories"] = parse_number(
        result.get("max_calories")
    )

    result["min_protein"] = parse_number(
        result.get("min_protein")
    )

    result["max_protein"] = parse_number(
        result.get("max_protein")
    )

    result["min_sodium"] = parse_number(
        result.get("min_sodium")
    )

    result["max_sodium"] = parse_number(
        result.get("max_sodium")
    )

    result["include_ingredients"] = normalize_list(
        result.get(
            "include_ingredients"
        )
    )

    result["exclude_ingredients"] = normalize_list(
        result.get(
            "exclude_ingredients"
        )
    )

    category = result.get(
        "category"
    )

    if category is not None:
        category = str(
            category
        ).strip()

        result["category"] = (
            category
            if category
            else None
        )

    menu_keyword = result.get(
        "menu_keyword"
    )

    if menu_keyword is not None:
        menu_keyword = str(
            menu_keyword
        ).strip()

        result["menu_keyword"] = (
            menu_keyword
            if menu_keyword
            else None
        )

    price_scope = result.get(
        "price_scope"
    )

    if price_scope not in [
        "menu",
        "meal",
    ]:
        result["price_scope"] = None

    slot = result.get(
        "slot"
    )

    valid_slots = {
        "RICE",
        "SOUP",
        "MAIN",
        "SIDE",
        "KIMCHI",
        "OTHER",
    }

    if slot:
        slot = str(
            slot
        ).upper()

    result["slot"] = (
        slot
        if slot in valid_slots
        else None
    )

    apply_numeric_regex(
        query,
        result,
    )

    print(
        "[PARSED QUERY]",
        result,
    )

    return result