from sqlalchemy import create_engine
import pandas as pd

class DBConnect():
    def __init__(self):
        self.config_79 = {
            'user':'postgres',
            'password':'qwerty12345*',
            'host':'localhost',
            'port':'5433',
            'database':'postgres'
        }

        self.connection_uri_79 = 'postgresql://{user}:{password}@{host}:{port}/{database}'.format(**self.config_79)

        self.cnx_79 = None
        self.query= None
        self.db_dataframe = None

    def db_connect(self):
        try:
            self.cnx_79 = create_engine(self.connection_uri_79)
            print(f'Connection Established for Server 79')

            self._test_connection()
        except Exception as e:
            print(f'Error Connecting to Database: {e} ')

    def _test_connection(self):        
        try:
            test2 = pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_schema = \'public\'", self.cnx_79)
            print(f'TestTable2 : Database Connected: PostgreSQL Connection Status : 202')
            del test2
        except Exception as e:
            print(f'Error retrieving data: {e}')
            if "MySQL server has gone away" in str(e) or "OperationalError" in str(e) or "psycopg2.errors" in str(e):
                print("SQL connection is unavailable. Please check your internet connection.")
                print(e)

    def users_conn(self):
        self.query = f"""
        SELECT * FROM local_db.users_sample
        """
        self.db_dataframe = pd.read_sql(self.query, self.cnx_79)
        return self.db_dataframe
    
    def close_connection(self):
        if self.cnx_79:
            self.cnx_79.dispose()
            print("Connection closed.")
