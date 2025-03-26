import os
import sqlite3

def get_db_path(db_name):
    """
    Builts full path to provided database name.

    Goes to the project's root directory and goes into the data folder in order
    to build out the path of the database relative to the root directory.

    Args:
        db_name (str): The name of the SQLite database file.

    Returns:
        db_path (str): The full path of the SQLite database file relative to the 
        project's root directory.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, 'data', db_name)

    return db_path

