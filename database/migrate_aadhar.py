import sqlite3

def migrate_database():
    conn = sqlite3.connect('retinopathy_agent.db')
    cursor = conn.cursor()
    
    # Drop existing tables
    cursor.execute("DROP TABLE IF EXISTS scans;")
    cursor.execute("DROP TABLE IF EXISTS patients;")
    
    # Create new patients table with Aadhar Number
    cursor.execute("""
    CREATE TABLE patients (
        aadhar_number TEXT PRIMARY KEY,
        o_id INTEGER NOT NULL,
        patient_name TEXT NOT NULL,
        FOREIGN KEY (o_id) REFERENCES ophthalmologists(o_id)
    );
    """)

    # Create new scans table
    cursor.execute("""
    CREATE TABLE scans (
        scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
        aadhar_number TEXT NOT NULL,
        scan_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        original_image_path TEXT NOT NULL,
        heatmap_path TEXT NOT NULL,
        ai_predicted_class INTEGER,
        ai_confidence REAL,
        doctor_verified_class INTEGER,
        is_ai_correct BOOLEAN,
        ready_for_retraining BOOLEAN DEFAULT 0,
        FOREIGN KEY (aadhar_number) REFERENCES patients(aadhar_number)
    );
    """)
    
    conn.commit()
    conn.close()
    print("Migration successful: patients and scans tables have been recreated with Aadhar Number as Primary Key.")

if __name__ == "__main__":
    migrate_database()
