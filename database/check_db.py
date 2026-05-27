import sqlite3

try:
    conn = sqlite3.connect('retinopathy_agent.db')
    cursor = conn.cursor()
    cursor.execute("SELECT o_id, name, email_id FROM ophthalmologists")
    users = cursor.fetchall()
    
    if len(users) == 0:
        print("DATABASE IS EMPTY: No doctors are registered!")
    else:
        print("--- REGISTERED DOCTORS ---")
        for user in users:
            print(f"ID: {user[0]} | Name: {user[1]} | Email: '{user[2]}'")
            
    conn.close()
except sqlite3.OperationalError:
    print("ERROR: Cannot find retinopathy_agent.db in this folder!")