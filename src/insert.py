import sqlite3
from .player_ids import get_player_ids


def insert_prep(df, db_path):
    """
    Prepares dataframe to be inserted into SQLite database.

    Takes a per game or advancded table dataframe and attaches the player id column from 
    the database then renames the dataframe columns to match the database column.

    Args:
        df (pd.DataFrame): A cleaned per game or advanced table DataFrame
        df_path (str): The SQLite database.

    Returns:
        df_copy (pd.DataFrame): A preped DataFrame for insertion into database.
    """

    df_copy = df.copy()

    df_copy['player_id'] = df_copy['Player'].apply(get_player_ids, db_path=db_path)

    id_col = df_copy.pop('player_id')
    df_copy.insert(0, 'player_id', id_col)
    df_copy.drop(['Player'], axis=1, inplace=True)

    df_columns = df_copy.columns
    
    replace_dict = {'3': '_3', '2': '_2', '%': '_PCT', '/': 'PER'}

    for key, val in replace_dict.items():
        df_columns = [ele.replace(key, val) for ele in df_columns]
        

    df_copy.columns = df_columns

    return df_copy


def insert(df, db_path, db_table):
    """
    Inserts dataframe values into the specified database table if it is not already there.

    Takes a dataframe and inserts the values with it into a specified database table, iterates through
    the rows and will insert into table if primary key does not exists. Handles pandas NAtypes in 'Salary'
    column by converting them into None.

    Args:
        df (pd.DataFrame): A preped dataframe that will be inserted to the database.
        df_path (str): The SQLite database.
        df_table (str): Specified table within datbase.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    df_columns = df.columns

    placeholders = ', '.join(['?'] * len(df_columns))
    insert_statement = f'INSERT OR IGNORE INTO {db_table} ({", ".join(df_columns)}) VALUES ({placeholders})'

    for _, row in df.iterrows():
        values = tuple(row.fillna({'Salary': None}))
        cursor.execute(insert_statement, values)
    
    conn.commit()
    conn.close()

