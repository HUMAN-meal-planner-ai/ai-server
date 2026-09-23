import csv
from pathlib import Path

# 메뉴 1개 정보 받아 rag 검색용 문서 1개 만드는 함수
def make_menu_document(
    menu_code: str,
    menu_name :str,
    main_category: str,
    sub_category: str,
) -> dict : #딕셔너리로 반환
    
    if not menu_code:
        raise ValueError("메뉴 코드가 없습니다.") # 메뉴 코드 비어 있을 경우 오류 메시지 띄우기
    
    return{
        "id": menu_code,
        "text": (
            f"메뉴명: {menu_name}\n"
            f"대분류: {main_category}\n"
            f"소분류: {sub_category}"
        ),
        "metadata": {
            "menu_code":menu_code
        }
    }
def load_menu_documents(csv_path: str | Path) -> list[dict]:

    # 만들어진 RAG 문서들을 차례로 담을 빈 리스트
    documents = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:

        # CSV의 첫 번째 줄을 컬럼명으로 사용해서 데이터를 읽음
        reader = csv.DictReader(file)

        # CSV에서 메뉴를 한 줄씩 꺼내 반복
        for row in reader:

            # 현재 메뉴의 정보를 함수에 전달해 RAG 문서 1개 생성
            document = make_menu_document(
                menu_code=row["menuCode"],           
                menu_name=row["menuName"],           
                main_category=row["mainCategory"],   
                sub_category=row["subCategory"],    
            )

            # 완성된 문서를 documents 리스트에 추가
            documents.append(document)

    # CSV의 모든 메뉴를 처리한 뒤 문서 목록 전체를 반환
    return documents
    
