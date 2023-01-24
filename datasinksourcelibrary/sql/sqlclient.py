import logging
import os
from urllib.parse import quote

import pandas as pd
from datasinksourcelibrary.sink_base import Sink
from datasinksourcelibrary.source_base import Source
from sqlalchemy import create_engine


class SQLClient:
    """Client to connect and interact with SQL server databases

    Args:
        connection_str (str): Connection string for connecting to Azure SQL.
            Defaults to None.
    """

    def __init__(self,
                 output_collector,
                 connection_str: str = None):
        self._logger = logging.getLogger(__name__)
        # check if missing connection information can be retrieved from env
        if connection_str is None:
            try:
                connection_str = os.environ["SQL_CONNECTION_STRING"]
            except ValueError as error:
                msg = f'{error}: Connection string not found in environment!'
                raise msg

        # Parse connection string for url
        connection_str = quote(connection_str)
        conn_string = f"mssql+pyodbc:///?odbc_connect={connection_str}"

        # Setup engine and connect
        self.engine = create_engine(
            conn_string,
            fast_executemany=True,
            connect_args={'connect_time': 20})


class SQLDataWriter(SQLClient, Sink):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # init the sink base
        self._out_coll = kwargs['output_collector']

    def upload_to_staging(self,
                          data: pd.DataFrame,
                          db_table: str,
                          db_schema: str = 'staging',
                          clear_table: bool = True):
        """ Upload data to staging table. First it removes all data in the
        table, and then insert data
        Args:
            engine (obj): SQLalchemy engine representation of DB
            data (pd.DataFrame): pandas dataframe of the data to upload
            db_table (str): table name in the DB to insert data
            db_schema (str, optional): schema of the table to insert data.
            Defaults to 'staging'.
            clear_table(bool, optional): flag defining whether table is cleared
            from data before data upload
            Defaults to True (=table is cleared)
        """

        # Delete all rows in DB table
        if clear_table:
            self.clear_table(db_schema=db_schema, db_table=db_table)

        with self.engine.connect() as conn:
            # write data to the DB table
            data.to_sql(
                db_table,
                schema=db_schema,
                con=conn,
                if_exists='append',
                index=False
            )
        return True

    # TODO: add more data types (perhaps convertion in other function)
    def write(self,
              data,
              data_type: str,
              db_table: str,
              db_schema: str = None,
              **kwargs):
        """Upload data to table. If table already exists, data is appended to
        existing table. If table does not exist, it is created.

        Args:
            data: data to upload to SQL
            data_type (str): data type to write from
            db_table (str): table name in the DB to insert data
            db_schema (str, optional): schema of the table. This should be a
            string of the schema in brackets. Defaults to None.

        """

        # Write depending on data type
        if data_type == 'dataframe':
            assert isinstance(data, pd.DataFrame), "Data incorrect type"
            # write data to the DB table
            with self.engine.connect() as conn:
                data.to_sql(
                    db_table,
                    schema=db_schema,
                    con=conn,
                    if_exists='append',
                    index=False
                )
        else:
            raise ValueError("Data type not supported")

        return True

    def run_stored_procedure(self, stored_procedure):
        """Calls the a stored procedure in a DB.
        Args:
            conn (SQLalchemy engine): engine connection to DB
            stored_procedure (str): name of the stored procedure to run,
            Should be in the format `[benchmarking].[usp_Weight]`
        """
        with self.engine.connect().execution_options(autocommit=True) as conn:
            # Run stored procedure in DB
            conn.execute(stored_procedure)

    def clear_table(self,
                    db_table: str,
                    db_schema: str):
        """clears a table in DB from all data

        Args:
            db_table (str): table name in the DB to insert data
            db_schema (str): schema of the table to insert data
        """
        with self.engine.connect().execution_options(autocommit=True) as conn:
            conn.execute(f'TRUNCATE TABLE [{db_schema}].[{db_table}]')


class SQLDataLoader(SQLClient, Source):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # init the source base
        self._out_coll = kwargs['output_collector']
        self.metadata = {}

    def load(self, query: str, query_type: str = None):
        """Implemented abstract method from Source base for loading data from
        database.

        Args:
            query (str): Query to get data with
            query_type (str): Query type. Defaults to None.

        Returns:
            df (pd.DataFrame): DataFrame with queried data.

        """

        # Get data as pd.DataFrame from SQL
        df = pd.read_sql(sql=query,
                         con=self.engine.connect())

        # sort DataFrame columns before returning
        col_names = list(df.columns).copy()
        col_names.sort(key=str)
        df = df[col_names]

        return df

    def _compose_query(self,
                       query_template: str,
                       query_config: dict):
        """Implementation of abstract method to compose query by populating
        template with key-values from config dict. Key is placeholder and
        value is replacement. Also reformats config to fit query.

        Args:
            query_template (str): Query template with placeholders.
            query_config (dict): Key-value pairs to populate template with.

        Returns:
            query (str): Populated template to use in query.
        """

        # copy config to modify it without changing the original object
        query_con = query_config.copy()

        # Join cols if in config, else SELECT *
        if "cols" in query_con.keys():
            query_con["cols"] = ', '.join(query_con["cols"])
        else:
            query_con["cols"] = '*'

        # Join conditions if exist, else remove WHERE of query
        if "conditions" in query_con.keys():
            query_con["conditions"] = ' AND '.join(query_con["conditions"])
        else:
            query_template = query_template.split(' WHERE')[0] + ';'

        # Populate template
        query = query_template
        for key, value in query_con.items():
            query = query.replace(f'[{key}]', value)

        return query
