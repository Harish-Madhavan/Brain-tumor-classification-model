import os
import sqlite3
import datetime
import pandas as pd

def init_db(db_path='patient_records.db'):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diagnostic_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                pred_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                tumor_area_cm2 REAL,
                tumor_width_cm REAL,
                tumor_height_cm REAL,
                report_path TEXT
            )
        """)
        
        # Check for missing patient info columns and migrate if necessary
        cursor.execute("PRAGMA table_info(diagnostic_records)")
        columns = [row[1] for row in cursor.fetchall()]
        
        new_cols = {
            'patient_name': 'TEXT',
            'patient_id': 'TEXT',
            'age': 'INTEGER',
            'gender': 'TEXT',
            'physician_notes': 'TEXT'
        }
        
        for col, col_type in new_cols.items():
            if col not in columns:
                cursor.execute(f"ALTER TABLE diagnostic_records ADD COLUMN {col} {col_type}")
        conn.commit()

def log_diagnostic_run(filename, pred_class, confidence, metrics, report_path=None, 
                       patient_name=None, patient_id=None, age=None, gender=None, physician_notes=None,
                       db_path='patient_records.db'):
    try:
        init_db(db_path)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            area_cm2 = metrics.get('total_area_cm2', 0.0) if metrics else 0.0
            width_cm = metrics.get('width_cm', 0.0) if metrics else 0.0
            height_cm = metrics.get('height_cm', 0.0) if metrics else 0.0
            
            cursor.execute("""
                INSERT INTO diagnostic_records 
                (timestamp, filename, pred_class, confidence, tumor_area_cm2, tumor_width_cm, tumor_height_cm, report_path,
                 patient_name, patient_id, age, gender, physician_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, filename, pred_class, confidence, area_cm2, width_cm, height_cm, report_path,
                  patient_name, patient_id, age, gender, physician_notes))
            conn.commit()
        print(f"Logged record for {filename} to local SQLite database.")
    except Exception as e:
        print(f"Error logging diagnostic run to DB: {e}")

def fetch_diagnostic_records(search_query=None, db_path='patient_records.db'):
    try:
        init_db(db_path)
        with sqlite3.connect(db_path) as conn:
            query = """
                SELECT id, timestamp, filename, patient_name, patient_id, age, gender, 
                       pred_class, confidence, tumor_area_cm2, tumor_width_cm, tumor_height_cm, 
                       physician_notes, report_path 
                FROM diagnostic_records
            """
            params = []
            if search_query:
                query += " WHERE filename LIKE ? OR pred_class LIKE ? OR patient_name LIKE ? OR patient_id LIKE ?"
                params = [f"%{search_query}%", f"%{search_query}%", f"%{search_query}%", f"%{search_query}%"]
                
            query += " ORDER BY id DESC"
            df = pd.read_sql_query(query, conn, params=params)
        return df
    except Exception as e:
        print(f"Error fetching diagnostic records: {e}")
        return pd.DataFrame()

def delete_diagnostic_record(record_id, db_path='patient_records.db'):
    try:
        init_db(db_path)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM diagnostic_records WHERE id = ?", (record_id,))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        print(f"Error deleting diagnostic record {record_id}: {e}")
        return False

def export_database_csv(db_path='patient_records.db', output_dir='classification_results'):
    try:
        df = fetch_diagnostic_records(db_path=db_path)
        os.makedirs(output_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join(output_dir, f"patient_database_export_{stamp}.csv")
        df.to_csv(export_path, index=False)
        return export_path
    except Exception as e:
        print(f"Error exporting database to CSV: {e}")
        return None

def get_database_stats(db_path='patient_records.db'):
    try:
        init_db(db_path)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM diagnostic_records")
            total_scans = cursor.fetchone()[0]
            
            cursor.execute("SELECT pred_class, COUNT(*) FROM diagnostic_records GROUP BY pred_class")
            class_counts = cursor.fetchall()
            class_dist = {cls: count for cls, count in class_counts}
            
            cursor.execute("SELECT AVG(tumor_area_cm2) FROM diagnostic_records WHERE tumor_area_cm2 > 0")
            avg_area = cursor.fetchone()[0]
            avg_area = avg_area if avg_area is not None else 0.0
            
            cursor.execute("SELECT AVG(confidence) FROM diagnostic_records")
            avg_conf = cursor.fetchone()[0]
            avg_conf = avg_conf if avg_conf is not None else 0.0
            
        return {
            'total_scans': total_scans,
            'class_dist': class_dist,
            'avg_area': avg_area,
            'avg_conf': avg_conf
        }
    except Exception as e:
        print(f"Error getting database stats: {e}")
        return {'total_scans': 0, 'class_dist': {}, 'avg_area': 0.0, 'avg_conf': 0.0}
