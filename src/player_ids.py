import sqlite3

def get_player_ids(name, db_path):
    """
    Checks if player has an associated player id in database.

    Checks if player has a player_id in players table, if player has an associated 
    player_id returns player_id, if not adds player to players table then generates 
    and return player_id.

    Args: 
        name (str): The name of the player.
        db_path (str): The database to in which to check for player id.
    
    Returns:
        player_id (int): The player's associated player id.

    Raises:
        sqlite3.Error: If an sqlite3 Error occurs during querying.
    """
    # connecting to database, queries for chosen player using parameterized query
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor() # creates cursor object to excecute queries
    cursor.execute('SELECT * from players WHERE player = ?', (name,)) # query for chosen player
    result = cursor.fetchone() # fetches the first result from query

    try:
        if result: # if there is a result 
            player_id = result[0] # result is the player_id
            return player_id
        else: # if result is none
            cursor.execute('SELECT MAX(player_id) FROM players') # looks for the highest player_id
            max_id = cursor.fetchone()[0] # fetches the highest/newest player_id
            player_id = 1 if max_id is None else max_id+1 # assigns first or next player_id

            # inserting player_id and name into players table, using parameterized query
            cursor.execute('INSERT INTO players (player_id, player) VALUES (?, ?)', (player_id, name))
            conn.commit()
            return player_id
    
    except sqlite3.Error as e: # handling errors
        print(f'Database Error: {e}')
        return None
    
    finally: # closing cursor and connection
        if cursor:
            cursor.close()
        if conn:
            conn.close()

