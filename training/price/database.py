from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg import Connection

from training.price.config import DatabaseSettings


@contextmanager
def open_read_only_connection(settings: DatabaseSettings) -> Iterator[Connection]:
    """학습 데이터 조회 과정에서 DB를 변경하지 못하도록 읽기 전용 연결을 연다."""
    with psycopg.connect(
        settings.url,
        user=settings.username,
        password=settings.password,
        options="-c default_transaction_read_only=on",
    ) as connection:
        yield connection
