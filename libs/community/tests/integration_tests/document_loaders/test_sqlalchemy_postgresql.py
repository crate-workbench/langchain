"""
Test SQLAlchemy/PostgreSQL document loader functionality.

cd tests/integration_tests/document_loaders/docker-compose
docker-compose -f postgresql.yml up
"""
import logging
import os
import unittest

import pytest
import sqlalchemy as sa
import sqlparse

from langchain.document_loaders.sqlalchemy import SQLAlchemyLoader
from tests.data import MLB_TEAMS_2012_SQL

logging.basicConfig(level=logging.DEBUG)


try:
    import psycopg2  # noqa: F401

    psycopg2_installed = True
except ImportError:
    psycopg2_installed = False


CONNECTION_STRING = os.environ.get(
    "TEST_POSTGRESQL_CONNECTION_STRING",
    "postgresql+psycopg2://postgres@localhost:5432/",
)


@pytest.fixture
def engine() -> sa.Engine:
    """
    Return an SQLAlchemy engine object.
    """
    return sa.create_engine(CONNECTION_STRING, echo=False)


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


@unittest.skipIf(not psycopg2_installed, "psycopg2 not installed")
def test_postgresql_loader_no_options() -> None:
    """Test SQLAlchemy loader with psycopg2."""

    loader = SQLAlchemyLoader("SELECT 1 AS a, 2 AS b", url=CONNECTION_STRING)
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "a: 1\nb: 2"
    assert docs[0].metadata == {}


@unittest.skipIf(not psycopg2_installed, "psycopg2 not installed")
def test_postgresql_loader_include_rownum_into_metadata() -> None:
    """Test SQLAlchemy loader with psycopg2."""

    loader = SQLAlchemyLoader(
        "SELECT 1 AS a, 2 AS b",
        url=CONNECTION_STRING,
        include_rownum_into_metadata=True,
    )
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "a: 1\nb: 2"
    assert docs[0].metadata == {"row": 0}


@unittest.skipIf(not psycopg2_installed, "psycopg2 not installed")
def test_postgresql_loader_include_query_into_metadata() -> None:
    """Test SQLAlchemy loader with psycopg2."""

    loader = SQLAlchemyLoader(
        "SELECT 1 AS a, 2 AS b", url=CONNECTION_STRING, include_query_into_metadata=True
    )
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "a: 1\nb: 2"
    assert docs[0].metadata == {"query": "SELECT 1 AS a, 2 AS b"}


@unittest.skipIf(not psycopg2_installed, "psycopg2 not installed")
def test_postgresql_loader_page_content_columns() -> None:
    """Test SQLAlchemy loader with psycopg2."""

    loader = SQLAlchemyLoader(
        "SELECT 1 AS a, 2 AS b UNION SELECT 3 AS a, 4 AS b",
        url=CONNECTION_STRING,
        page_content_columns=["a"],
    )
    docs = loader.load()

    assert len(docs) == 2
    assert docs[0].page_content == "a: 1"
    assert docs[0].metadata == {}

    assert docs[1].page_content == "a: 3"
    assert docs[1].metadata == {}


@unittest.skipIf(not psycopg2_installed, "psycopg2 not installed")
def test_postgresql_loader_metadata_columns() -> None:
    """Test SQLAlchemy loader with psycopg2."""

    loader = SQLAlchemyLoader(
        "SELECT 1 AS a, 2 AS b",
        url=CONNECTION_STRING,
        page_content_columns=["a"],
        metadata_columns=["b"],
    )
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "a: 1"
    assert docs[0].metadata == {"b": 2}


@unittest.skipIf(not psycopg2_installed, "psycopg2 not installed")
def test_postgresql_loader_real_data_with_sql(provision_database: None) -> None:
    """Test SQLAlchemy loader with psycopg2."""

    loader = SQLAlchemyLoader(
        query='SELECT * FROM mlb_teams_2012 ORDER BY "Team";',
        url=CONNECTION_STRING,
    )
    docs = loader.load()

    assert len(docs) == 30
    assert docs[0].page_content == "Team: Angels\nPayroll (millions): 154.49\nWins: 89"
    assert docs[0].metadata == {}


@unittest.skipIf(not psycopg2_installed, "psycopg2 not installed")
def test_postgresql_loader_real_data_with_selectable(provision_database: None) -> None:
    """Test SQLAlchemy loader with psycopg2."""

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
        url=CONNECTION_STRING,
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
