from mcp.server import MCPServer


mcp = MCPServer("Mealfit MCP")


@mcp.tool()
def hello(name: str) -> str:
    return f"안녕하세요, {name}"


if __name__ == "__main__":
    mcp.run()