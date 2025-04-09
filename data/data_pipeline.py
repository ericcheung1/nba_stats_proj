import sys
import os

# Check if __file__ is defined (running from a file)
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # If __file__ is not defined (running in REPL or code cell), use os.getcwd()
    project_root = os.getcwd()

sys.path.append(project_root)
from src.scraper2 import scrape_page
from src.cleaner import clean_bbref, clean_salary, salary_match
from src.insert import insert_prep, insert
from src.get_db_path import get_db_path
import pandas as pd

Per_Game_Page = 'https://www.basketball-reference.com/leagues/NBA_2021_per_game.html'
Advanced_Page = 'https://www.basketball-reference.com/leagues/NBA_2021_advanced.html'
Salary_Page = 'https://hoopshype.com/salaries/players/2020-2021/'
Database_name = 'nba_stats.db'

def pipeline(url1, url2, url3, db_name):
    """"
    Executes the pipeline and scrapes, clean, and loads the data.

    Takes a per game page, advanced page from Basketball-Reference and a 
    salary page from HoopsHype. Merges the salary and per game tables and 
    inserts the merged and advanced tables into a SQLite database.

    Args:
        url1 (str): URL of the Per Game page.
        url2 (str): URL of the Advanced page.
        url3 (str): URL of the Salary page.
        db_name (str): SQLite Database name.
    
    Returns:
        
        DataFrame (pd.Dataframe): A printout of the merged and advanced tables joined together.
    """
    db_path = get_db_path(db_name)

    scraped_Per_Game = scrape_page(url1)
    scraped_Advanced = scrape_page(url2)
    scraped_Salary = scrape_page(url3)

    cleaned_Per_Game = clean_bbref(scraped_Per_Game)
    cleaned_Advanced = clean_bbref(scraped_Advanced)
    cleaned_salary = clean_salary(scraped_Salary)

    cleaned_PerGame_Salary = salary_match(cleaned_Per_Game, cleaned_salary)

    preped_PGS = insert_prep(cleaned_PerGame_Salary, db_path)
    preped_Advanced = insert_prep(cleaned_Advanced, db_path)

    insert(preped_PGS, db_path, 'PerGameBasic')
    insert(preped_Advanced, db_path, 'PerGameAdvanced')

    df = pd.merge(cleaned_PerGame_Salary, cleaned_Advanced, how='left', on=['Player', 'G', 'GS', 'Age', 'Pos', 'Team', 'season'])  # type: ignore
    return df.head()

print(pipeline(Per_Game_Page, Advanced_Page, Salary_Page, Database_name))

