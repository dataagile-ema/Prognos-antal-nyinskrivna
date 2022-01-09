import pandas as pd

def read_and_prep_data(path: str):
    """
    läser csv fil med format ds (datum) och y (antal nyinskrivna)
    """
    df = pd.read_csv(path)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values(by='ds')
    return df