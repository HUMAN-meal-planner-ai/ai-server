from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class DatabaseSettings:
    """가격 데이터 읽기에 필요한 PostgreSQL 연결 설정."""

    url: str
    username: str
    password: str

    @classmethod
    def from_environment(cls, env_file: Path | None = None) -> "DatabaseSettings":
        load_dotenv(dotenv_path=env_file, override=False)
        values = {
            "DB_URL": os.getenv("DB_URL"),
            "DB_USERNAME": os.getenv("DB_USERNAME"),
            "DB_PASSWORD": os.getenv("DB_PASSWORD"),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise RuntimeError(
                "PostgreSQL 연결 환경변수가 필요합니다: " + ", ".join(missing)
            )

        return cls(
            url=_normalize_postgresql_url(values["DB_URL"]),
            username=values["DB_USERNAME"],
            password=values["DB_PASSWORD"],
        )


def _normalize_postgresql_url(url: str) -> str:
    """backend와 공유하는 JDBC URL을 psycopg가 이해하는 URL로 변환한다."""
    normalized = url.strip()
    if normalized.startswith("jdbc:"):
        normalized = normalized.removeprefix("jdbc:")
    if not normalized.startswith(("postgresql://", "postgres://")):
        raise ValueError("DB_URL은 PostgreSQL 연결 URL이어야 합니다.")
    return normalized
