import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # ai-server/.env에서 OPENAI_API_KEY 읽기
client = OpenAI()


def parse_query(question: str) -> dict:
    # LLM에게 질문에서 조건만 추출하도록 요청
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "메뉴·식단 검색 질문에서 조건만 추출해 JSON으로 답하세요. "
                    "키는 max_price, category, price_scope를 사용하세요. "
                    "max_price는 최대 금액(원), 없으면 null. "
                    "category는 질문에 명시된 음식 분류, 없으면 null. "
                    "price_scope는 메뉴 하나면 'menu', 식단 전체면 'meal', "
                    "불분명하면 null. 없는 조건은 추측하지 마세요."
                ),
            },
            {"role": "user", "content": question},
        ],
    )

    return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    question = input("질문을 입력하세요: ")
    print(parse_query(question))