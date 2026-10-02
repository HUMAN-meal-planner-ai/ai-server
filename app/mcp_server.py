from mcp.server import MCPServer

from app.rag.search_menus import search_menus


mcp = MCPServer("Mealfit MCP")


# 조건에 맞는 급식 메뉴 검색
@mcp.tool()
def search_menu(
    query: str,
    authorization: str | None = None,
    conditions: dict | None = None,
) -> list[dict]:
    """조건에 맞는 급식 메뉴를 검색합니다."""

    print("[MCP SERVER] search_menu 호출")
    print("[MCP SERVER] query:", query)
    print("[MCP SERVER] conditions:", conditions)

    menus = search_menus(
        query=query,
        authorization=authorization,
        conditions=conditions,
    )

    print("[MCP SERVER] 검색 결과 수:", len(menus))
    print("[MCP SERVER] 검색 결과 타입:", type(menus))

    return menus


if __name__ == "__main__":
    mcp.run()