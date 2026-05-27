import sqlite3

def setup_database():
    conn = sqlite3.connect('retinopathy_agent.db')
    cursor = conn.cursor()
    
    # 1. The Ophthalmologist Table (For Auth and Isolation)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ophthalmologists (
        o_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email_id TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        dob TEXT,
        working_address TEXT
    );
    """)
    
    # 2. The Patient Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        p_id INTEGER PRIMARY KEY AUTOINCREMENT,
        o_id INTEGER NOT NULL,
        patient_name TEXT NOT NULL,
        FOREIGN KEY (o_id) REFERENCES ophthalmologists(o_id)
    );
    """)

    # 3. The Scans Table (Where the AI and Doctor interact)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scans (
        scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
        p_id INTEGER NOT NULL,
        scan_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        original_image_path TEXT NOT NULL,
        heatmap_path TEXT NOT NULL,
        ai_predicted_class INTEGER,
        ai_confidence REAL,
        doctor_verified_class INTEGER,  -- NULL until the doctor reviews it
        is_ai_correct BOOLEAN,          -- True/False
        ready_for_retraining BOOLEAN DEFAULT 0,
        FOREIGN KEY (p_id) REFERENCES patients(p_id)
    );
    """)
    
    conn.commit()
    conn.close()
    print("Clinical Database successfully built with relational tables!")

if __name__ == "__main__":
    setup_database()