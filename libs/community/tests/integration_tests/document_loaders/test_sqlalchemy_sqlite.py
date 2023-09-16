"""
Test SQLAlchemy/SQLite document loader functionality.
"""
import logging
import unittest

import pytest
import sqlalchemy as sa
import sqlparse
from _pytest.tmpdir import TempPathFactory

from langchain.document_loaders.sqlalchemy import SQLAlchemyLoader
from tests.data import MLB_TEAMS_2012_SQL

logging.basicConfig(level=logging.DEBUG)


try:
    import sqlite3  # noqa: F401

    sqlite3_installed = True
except ImportError:
    sqlite3_installed = False


@pytest.fixture(scope="module")
def db_uri(tmp_path_factory: TempPathFactory) -> str:
    """
    Return an SQLAlchemy URI for a temporary SQLite database.
    """
    db_path = tmp_path_factory.getbasetemp().joinpath("testdrive.sqlite")
    return f"sqlite:///{db_path}"


@pytest.fixture(scope="module")
def engine(db_uri: str) -> sa.Engine:
    """
    Return an SQLAlchemy engine object.
    """
    return sa.create_engine(db_uri, echo=False)


@pytest.fixture()
def provision_database(engine: sa.Engine) -> None:
    """
    Provision database with table schema and data.
    """
    sql_statements = MLB_TEAMS_2012_SQL.read_text()
    with engine.connect() as connection:
        connection.execute(sa.text("DROP TABLE IF EXISTS mlb_teams_2012;"))
        for statement in sqlparse.split(sql_statements):
            connection.execute(sa.text(statement))
            connection.commit()


@unittest.skipIf(not sqlite3_installed, "sqlite3 not installed")
def test_sqlite_loader_no_options(db_uri: str) -> None:
    """Test SQLAlchemy loader with sqlite3."""

    loader = SQLAlchemyLoader("SELECT 1 AS a, 2 AS b", url=db_uri)
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "a: 1\nb: 2"
    assert docs[0].metadata == {}


@unittest.skipIf(not sqlite3_installed, "sqlite3 not installed")
def test_sqlite_loader_include_rownum_into_metadata(db_uri: str) -> None:
    """Test SQLAlchemy loader with sqlite3."""

    loader = SQLAlchemyLoader(
        "SELECT 1 AS a, 2 AS b",
        url=db_uri,
        include_rownum_into_metadata=True,
    )
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "a: 1\nb: 2"
    assert docs[0].metadata == {"row": 0}


@unittest.skipIf(not sqlite3_installed, "sqlite3 not installed")
def test_sqlite_loader_include_query_into_metadata(db_uri: str) -> None:
    """Test SQLAlchemy loader with sqlite3."""

    loader = SQLAlchemyLoader(
        "SELECT 1 AS a, 2 AS b", url=db_uri, include_query_into_metadata=True
    )
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "a: 1\nb: 2"
    assert docs[0].metadata == {"query": "SELECT 1 AS a, 2 AS b"}


@unittest.skipIf(not sqlite3_installed, "sqlite3 not installed")
def test_sqlite_loader_page_content_columns(db_uri: str) -> None:
    """Test SQLAlchemy loader with sqlite3."""

    loader = SQLAlchemyLoader(
        "SELECT 1 AS a, 2 AS b UNION SELECT 3 AS a, 4 AS b",
        url=db_uri,
        page_content_columns=["a"],
    )
    docs = loader.load()

    assert len(docs) == 2
    assert docs[0].page_content == "a: 1"
    assert docs[0].metadata == {}

    assert docs[1].page_content == "a: 3"
    assert docs[1].metadata == {}


@unittest.skipIf(not sqlite3_installed, "sqlite3 not installed")
def test_sqlite_loader_metadata_columns(db_uri: str) -> None:
    """Test SQLAlchemy loader with sqlite3."""

    loader = SQLAlchemyLoader(
        "SELECT 1 AS a, 2 AS b",
        url=db_uri,
        page_content_columns=["a"],
        metadata_columns=["b"],
    )
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "a: 1"
    assert docs[0].metadata == {"b": 2}


@unittest.skipIf(not sqlite3_installed, "sqlite3 not installed")
def test_sqlite_loader_real_data_with_sql(
    db_uri: str, provision_database: None
) -> None:
    """Test SQLAlchemy loader with sqlite3."""

    loader = SQLAlchemyLoader(
        query='SELECT * FROM mlb_teams_2012 ORDER BY "Team";',
        url=db_uri,
    )
    docs = loader.load()

    assert len(docs) == 30
    assert docs[0].page_content == "Team: Angels\nPayroll (millions): 154.49\nWins: 89"
    assert docs[0].metadata == {}


@unittest.skipIf(not sqlite3_installed, "sqlite3 not installed")
def test_sqlite_loader_real_data_with_selectable(
    db_uri: str, provision_database: None
) -> None:
    """Test SQLAlchemy loader with sqlite3."""

    # Define an SQLAlchemy table.
    mlb_teams_2012 = sa.Table(
        "mlb_teams_2012",
        sa.MetaData(),
        sa.Column("Team", sa.VARCHAR),
        sa.Column("Payroll (millions)", sa.FLOAT),
        sa.Column("Wins", sa.BIGINT),
    )

    # Query the database table using an SQLAlchemy selectable.
    select = sa.select(mlb_teams_2012).order_by(mlb_teams_2012.c.Team)
    loader = SQLAlchemyLoader(
        query=select,
        url=db_uri,
        include_query_into_metadata=True,
    )
    docs = loader.load()

    assert len(docs) == 30
    assert docs[0].page_content == "Team: Angels\nPayroll (millions): 154.49\nWins: 89"
    assert docs[0].metadata == {
        "query": 'SELECT mlb_teams_2012."Team", mlb_teams_2012."Payroll (millions)", '
        'mlb_teams_2012."Wins" \nFROM mlb_teams_2012 '
        'ORDER BY mlb_teams_2012."Team"'
    }
