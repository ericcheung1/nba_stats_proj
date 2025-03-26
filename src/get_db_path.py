import os
import sqlite3

def get_db_path(db_name):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, 'data', db_name)

    return db_path

