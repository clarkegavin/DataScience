import pandas as pd
from sqlalchemy import create_engine


def connect():
    eng = create_engine('mssql+pyodbc://GAVIN-PC\SQLEXPRESS01/acled?driver=SQL+Server+Native+Client+11.0')
    return eng


def get_top100_data(engine):
    data = pd.read_sql("SELECT TOP 100 * FROM acled", engine)
    return data

def get_all_data(engine):
    data = pd.read_sql("SELECT * FROM acled", engine)
    return data

def dispose_engine(engine):
    engine.dispose()


if __name__ == '__main__':
    engine = connect()
    df = get_top100_data(engine)
    dispose_engine(engine)
