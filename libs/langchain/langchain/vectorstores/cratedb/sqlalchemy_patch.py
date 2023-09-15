# mypy: disable-error-code=no-untyped-def
import itertools
import typing as t

import sqlalchemy as sa
from sqlalchemy.event import listen


def polyfill_refresh_after_dml(session):
    """
    Run `REFRESH TABLE <tablename>` after each INSERT, UPDATE, and DELETE operation.

    CrateDB is eventually consistent, i.e. write operations are not flushed to
    disk immediately, so readers may see stale data. In a traditional OLTP-like
    application, this is not applicable.

    This SQLAlchemy extension makes sure that data is synchronized after each
    operation manipulating data.

    > `after_{insert,update,delete}` events only apply to the session flush operation
    > and do not apply to the ORM DML operations described at ORM-Enabled INSERT,
    > UPDATE, and DELETE statements. To intercept ORM DML events, use
    > `SessionEvents.do_orm_execute().`
    > -- https://docs.sqlalchemy.org/en/20/orm/events.html#sqlalchemy.orm.MapperEvents.after_insert

    > Intercept statement executions that occur on behalf of an ORM Session object.
    > -- https://docs.sqlalchemy.org/en/20/orm/events.html#sqlalchemy.orm.SessionEvents.do_orm_execute

    > Execute after flush has completed, but before commit has been called.
    > -- https://docs.sqlalchemy.org/en/20/orm/events.html#sqlalchemy.orm.SessionEvents.after_flush

    TODO: Submit patch to `crate-python`, to be enabled by a
          dialect parameter `crate_dml_refresh` or such.
    """  # noqa: E501
    listen(session, "after_flush", do_flush)


def do_flush(session, flush_context):
    """
    SQLAlchemy event handler for the 'after_flush' event,
    invoking `REFRESH TABLE` on each table which has been modified.
    """
    dirty_entities = itertools.chain(session.new, session.dirty, session.deleted)
    dirty_classes = set([entity.__class__ for entity in dirty_entities])
    for class_ in dirty_classes:
        refresh_table(session, class_)


def refresh_table(connection, target):
    """
    Invoke a `REFRESH TABLE` statement.
    """
    sql = f"REFRESH TABLE {target.__tablename__}"
    connection.execute(sa.text(sql))


def patch_sqlalchemy_inspector():
    """
    When using `get_table_names()`, make sure the correct schema name gets used.

    Apparently, SQLAlchemy does not honor the `search_path` of the engine, when
    using the inspector?

    FIXME: Bug in CrateDB SQLAlchemy dialect?
    """

    def get_effective_schema(engine: sa.Engine):
        schema_name_raw = engine.url.query.get("schema")
        schema_name = None
        if isinstance(schema_name_raw, str):
            schema_name = schema_name_raw
        elif isinstance(schema_name_raw, tuple):
            schema_name = schema_name_raw[0]
        return schema_name

    from crate.client.sqlalchemy.dialect import CrateDialect

    get_table_names_dist = CrateDialect.get_table_names

    def get_table_names(
        self, connection: sa.Connection, schema: t.Optional[str] = None, **kw: t.Any
    ) -> t.List[str]:
        if schema is None:
            schema = get_effective_schema(connection.engine)
        return get_table_names_dist(self, connection=connection, schema=schema, **kw)

    CrateDialect.get_table_names = get_table_names  # type: ignore


def patch_sqlalchemy_ddl_compiler():
    """
    Ignore foreign key constraints when generating DDL statements,
    CrateDB does not understand them.

    FIXME: Needs to be added to the CrateDB SQLAlchemy dialect.
    """
    from crate.client.sqlalchemy.compiler import CrateDDLCompiler

    def visit_foreign_key_constraint(self, constraint, **kw):
        return None

    CrateDDLCompiler.visit_foreign_key_constraint = visit_foreign_key_constraint
