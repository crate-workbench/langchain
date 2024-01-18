import sqlalchemy as sa


def polyfill_refresh_after_dml_engine(engine: sa.engine.Engine):
    def receive_after_execute(
        conn: sa.engine.Connection,
        clauseelement,
        multiparams,
        params,
        execution_options,
        result,
    ):
        """
        Run a `REFRESH TABLE ...` command after each DML operation (INSERT, UPDATE,
        DELETE). This is used by CrateDB's Singer/Meltano and `rdflib-sqlalchemy`
        adapters.

        TODO: Pull in from a future `sqlalchemy-cratedb`.
        """
        if isinstance(clauseelement, (sa.sql.Insert, sa.sql.Update, sa.sql.Delete)):
            if not isinstance(clauseelement.table, sa.sql.Join):
                full_table_name = f'"{clauseelement.table.name}"'
                if clauseelement.table.schema is not None:
                    full_table_name = (
                        f'"{clauseelement.table.schema}".' + full_table_name
                    )
                conn.execute(sa.text(f"REFRESH TABLE {full_table_name};"))

    sa.event.listen(engine, "after_execute", receive_after_execute)
