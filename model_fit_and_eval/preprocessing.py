import sys
import os
import pandas as pd
import numpy as np
import sqlite3
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from src.get_db_path import get_db_path

db_path = get_db_path('nba_stats.db')
seasons = ['2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24'] # list of seasons to query from database
test_season = 'season_2023-24' # format as "season_" + Test Season

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

    merge_list = ['player', 'player_id','G', 'GS', 'Age', 'Pos', 'Team', 'season']
    Advanced['player_id'] = Advanced['player_id'].apply(pd.to_numeric, errors='coerce').astype('Int64')
    return pd.merge(PerGame, Advanced, how='left', on=merge_list)

list_of_seasons = []
for season in seasons:
    data = get_data(db_path, season)
    list_of_seasons.append(data)
full_data = pd.concat(list_of_seasons)

Salary_caps = pd.read_csv(os.path.join(project_root, 'data', 'cap_hist.csv'))
cap_dict = Salary_caps.set_index('season')['Salary_cap'].to_dict()

full_data['Salary_cap'] = full_data['season'].apply(lambda x: cap_dict.get(x))
full_data['Cap_Pct'] = np.round((full_data['Salary']/full_data['Salary_cap']), 2)
full_data.dropna(inplace=True)

full_data.sort_values(['season', 'player_id'], inplace=True)
full_data_wd = pd.get_dummies(full_data, columns=['Pos', 'season', 'Team'], drop_first=True, dtype=int)
train_data_wd = full_data_wd.loc[full_data_wd[test_season] == 0]
test_data_wd = full_data_wd.loc[full_data_wd[test_season] == 1]

full_data_wd.to_csv(os.path.join(project_root, 'data', 'full_data_six_seasons.csv'), index=False)
print(f"Matching Columns: {sum(test_data_wd.columns == train_data_wd.columns)}")
print(f"Full Data Shape: {full_data_wd.shape}")
print(f"Train Data Shape: {train_data_wd.shape}")
print(f"Test Data Shape: {test_data_wd.shape}")