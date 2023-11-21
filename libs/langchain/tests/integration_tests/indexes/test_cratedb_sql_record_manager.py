import os
import typing as t

import pytest
import sqlalchemy as sa

from langchain.indexes._sql_record_manager import Base, SQLRecordManager
from langchain.vectorstores.cratedb import CrateDBVectorSearch

CONNECTION_STRING = CrateDBVectorSearch.connection_string_from_db_params(
    driver=os.environ.get("TEST_CRATEDB_DRIVER", "crate"),
    host=os.environ.get("TEST_CRATEDB_HOST", "localhost"),
    port=int(os.environ.get("TEST_CRATEDB_PORT", "4200")),
    database=os.environ.get("TEST_CRATEDB_DATABASE", "testdrive"),
    user=os.environ.get("TEST_CRATEDB_USER", "crate"),
    password=os.environ.get("TEST_CRATEDB_PASSWORD", ""),
)


@pytest.fixture
def engine() -> sa.Engine:
    """
    Return an SQLAlchemy engine object.
    """
    return sa.create_engine(CONNECTION_STRING, echo=False)


@pytest.fixture(scope="session", autouse=True)
def dialect_patch_session(session_mocker: t.Any) -> None:
    """
    Patch the CrateDB SQLAlchemy dialect to ignore INDEX constraints.
    """
    import warnings

    from crate.client.sqlalchemy.compiler import CrateDDLCompiler

    def visit_create_index(
        self: t.Type[CrateDDLCompiler], *args: t.List, **kwargs: t.Dict
    ) -> str:
        """
        CrateDB does not support index constraints.

        CREATE INDEX ix_upsertion_record_group_id ON upsertion_record (group_id)
        """
        warnings.warn(
            "CrateDB does not support index constraints, "
            "they will be omitted when generating DDL statements."
        )
        return "SELECT 1;"

    session_mocker.patch(
        "crate.client.sqlalchemy.compiler.CrateDDLCompiler.visit_create_index",
        visit_create_index,
    )


@pytest.fixture(autouse=True)
def dialect_patch_function(monkeypatch: t.Any) -> None:
    """
    Patch the CrateDB SQLAlchemy dialect to handle `INSERT ... ON CONFLICT`
    operations like PostgreSQL.
    """
    from crate.client.sqlalchemy.compiler import CrateCompiler
    from sqlalchemy.dialects.postgresql.base import PGCompiler

    monkeypatch.setattr(
        CrateCompiler,
        "_on_conflict_target",
        PGCompiler._on_conflict_target,
        raising=False,
    )
    monkeypatch.setattr(
        CrateCompiler,
        "visit_on_conflict_do_nothing",
        PGCompiler.visit_on_conflict_do_nothing,
        raising=False,
    )
    monkeypatch.setattr(
        CrateCompiler,
        "visit_on_conflict_do_update",
        PGCompiler.visit_on_conflict_do_update,
        raising=False,
    )


@pytest.fixture(autouse=True)
def drop_tables(engine: sa.Engine) -> None:
    """
    Drop database tables before invoking test case function.
    """
    try:
        Base.metadata.drop_all(engine, checkfirst=False)
    except Exception as ex:
        if "RelationUnknown" not in str(ex):
            raise


@pytest.fixture()
def manager() -> SQLRecordManager:
    """Initialize the test database and yield the TimestampedSet instance."""
    # Initialize and yield the TimestampedSet instance
    record_manager = SQLRecordManager("kittens", db_url=CONNECTION_STRING)
    record_manager.create_schema()
    return record_manager


def test_update(manager: SQLRecordManager) -> None:
    """Test updating records in the database."""
    # no keys should be present in the set
    read_keys = manager.list_keys()
    assert read_keys == []
    # Insert records
    keys = ["key1", "key2", "key3"]
    manager.update(keys)
    # Retrieve the records
    read_keys = manager.list_keys()
    assert read_keys == ["key1", "key2", "key3"]
