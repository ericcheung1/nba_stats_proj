import sys
import os

# Check if __file__ is defined (running from a file)
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # If __file__ is not defined (running in REPL or code cell), use os.getcwd()
    project_root = os.getcwd()

sys.path.append(project_root)

from src.get_db_path import get_db_path
import pandas as pd
import numpy as np
import sqlite3

db_path = get_db_path('nba_stats.db')
merge_list = ['player', 'player_id','G', 'GS', 'Age', 'Pos', 'Team', 'season']

def get_data(db_path, season):

    conn = sqlite3.Connection(db_path)
    PG24 = f"""
    SELECT p.player, b.* 
    FROM players p 
    JOIN PerGameBasic b 
    ON p.player_id = b.player_id
    WHERE season = '{season}'
    """
    PerGame = pd.read_sql(PG24, conn)

    AD24 = f"""
    SELECT p.player, a.* 
    FROM players p
    JOIN PerGameAdvanced a
    ON p.player_id = a.player_id
    WHERE season = '{season}'
    """
    Advanced = pd.read_sql(AD24, conn)
    conn.close()

    Advanced['player_id'] = Advanced['player_id'].apply(pd.to_numeric, errors='coerce').astype('Int64')
    return pd.merge(PerGame, Advanced, how='left', on=merge_list)

_2021_22 = get_data(db_path, '2021-22')
_2022_23 = get_data(db_path, '2022-23')
_2023_24 = get_data(db_path, '2023-24')

data = pd.concat([_2021_22, _2022_23, _2023_24])
Salary_caps = {'2021-22': 112414000,'2022-23': 123655000,'2023-24': 136021000}
data['Salary_cap'] = data['season'].apply(lambda x: Salary_caps.get(x))
data['Cap_Pct'] = np.round((data['Salary']/data['Salary_cap']), 2)
data.dropna(inplace=True)

# PerGame['Salary_Cap'] = 136021000
# PerGame['Cap_Pct'] = np.round((PerGame['Salary']/PerGame['Salary_Cap'])*100, 2)
# PerGame.dropna(inplace=True)
# print(PerGame)