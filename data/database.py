import sqlite3

conn = sqlite3.connect('nba_stats.db')
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS players (
               player_id INTEGER PRIMARY KEY,
               player TEXT
    )
""")
conn.commit()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS PerGameBasic (
            player_id INTEGER,
            Salary INTEGER,
            Age INTEGER,
            Team TEXT,
            Pos TEXT,
            G INTEGER,
            GS INTEGER,
            MP REAL,
            FG REAL,
            FGA REAL,
            FG_PCT REAL,
            _3P REAL,
            _3PA REAL,
            _3P_PCT REAL,
            _2P REAL,
            _2PA REAL,
            _2P_PCT REAL,
            eFG_PCT REAL,
            FT REAL,
            FTA REAL,
            FT_PCT REAL,
            ORB REAL,
            DRB REAL,
            TRB REAL,
            AST REAL,
            STL REAL, 
            BLK REAL,
            TOV REAL,
            PF REAL,
            PTS REAL,
            season TEXT,
            PRIMARY KEY (player_id, Team, season),
            FOREIGN KEY (player_id)
            REFERENCES players (player_id)
        )
""")
conn.commit()

conn.execute("""
    CREATE TABLE IF NOT EXISTS PerGameAdvanced (
            player_id TEXT,
            Age INTEGER,
            Team TEXT,
            Pos TEXT,
            G INTEGER,
            GS INTEGER,
            TMP INTEGER,
            PER REAL,
            TS_PCT REAL,
            _3PAr REAL,
            FTr REAL,
            ORB_PCT REAL,
            DRB_PCT REAL,
            TRB_PCT REAL,
            AST_PCT REAL,
            STL_PCT REAL,
            BLK_PCT REAL,
            TOV_PCT REAL,
            USG_PCT REAL,
            OWS REAL,
            DWS REAL,
            WS REAL,
            WSPER48 REAL,
            OBPM REAL,
            DBPM REAL,
            BPM REAL,
            VORP REAL,
            season TEXT,
            PRIMARY KEY (player_id, Team, season),
            FOREIGN KEY (player_id)
            REFERENCES players (player_id)
            )
""")
conn.commit()