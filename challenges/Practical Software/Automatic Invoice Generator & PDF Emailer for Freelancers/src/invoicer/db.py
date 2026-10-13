"""SQLite engine and sessions."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, event
from sqlmodel import Session, SQLModel, create_engine

from invoicer import models  # noqa: F401  (registers the tables on SQLModel.metadata)


def make_engine(path: Path | str) -> Engine:
    """``path`` is a file, or ``:memory:``-style SQLAlchemy URL when it starts with ``sqlite``."""
    url = (
        str(path)
        if str(path).startswith("sqlite")
        else f"sqlite:///{Path(path).as_posix()}"
    )
    engine = create_engine(url, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute(
            "PRAGMA foreign_keys=ON"
        )  # SQLite ignores foreign keys unless asked
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()

    return engine


def init_db(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)


@contextmanager
def open_session(engine: Engine) -> Iterator[Session]:
    """Commit on success, roll back on any exception."""
    with Session(engine, expire_on_commit=False) as session:
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
