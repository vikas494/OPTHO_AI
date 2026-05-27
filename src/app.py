import streamlit as st
import os
import sqlite3
import hashlib
import time
import shutil
from dr_agent import DRAgent

st.set_page_config(page_title="VIK's Acadamy - DR AI", page_icon="👁️", layout="wide")

# --- BUG FIX 2: ABSOLUTE PATHS ---
# This forces the app to ALWAYS find the correct database and image folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, "../database/retinopathy_agent.db"))
SCANS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../data/scans"))
HEATMAPS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../data/heatmaps"))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../models"))

os.makedirs(SCANS_DIR, exist_ok=True)
os.makedirs(HEATMAPS_DIR, exist_ok=True)

# --- AUTHENTICATION HELPERS ---
def hash_password(password):
    return hashlib.sha256(password.strip().encode()).hexdigest()

def register_doctor(name, email, password, dob, address):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO ophthalmologists (name, email_id, password_hash, dob, working_address) VALUES (?, ?, ?, ?, ?)", 
                       (name, email.strip().lower(), hash_password(password), str(dob), address))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def login_doctor(email, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT o_id, name FROM ophthalmologists WHERE email_id=? AND password_hash=?", 
                   (email.strip().lower(), hash_password(password)))
    user = cursor.fetchone()
    conn.close()
    return user

# --- CLINICAL DATABASE HELPERS ---
def save_clinical_record(o_id, patient_name, aadhar_number, scan_path, heatmap_path, dr_class, confidence_str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT aadhar_number FROM patients WHERE aadhar_number=?", (aadhar_number,))
    patient = cursor.fetchone()
    if not patient:
        cursor.execute("INSERT INTO patients (aadhar_number, o_id, patient_name) VALUES (?, ?, ?)", (aadhar_number, o_id, patient_name))
        
    conf_decimal = float(confidence_str.strip('%')) / 100.0
    
    cursor.execute("""
        INSERT INTO scans (aadhar_number, original_image_path, heatmap_path, ai_predicted_class, ai_confidence)
        VALUES (?, ?, ?, ?, ?)
    """, (aadhar_number, scan_path, heatmap_path, dr_class, conf_decimal))
    conn.commit()
    conn.close()

def get_doctor_history(o_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.scan_id, p.patient_name, s.scan_date, s.original_image_path, s.heatmap_path, 
               s.ai_predicted_class, s.ai_confidence, s.doctor_verified_class, p.aadhar_number
        FROM scans s
        JOIN patients p ON s.aadhar_number = p.aadhar_number
        WHERE p.o_id = ?
        ORDER BY s.scan_date DESC
    """, (o_id,))
    history = cursor.fetchall()
    conn.close()
    return history

def get_unverified_scans(o_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.scan_id, p.patient_name, s.scan_date, s.original_image_path, s.heatmap_path, 
               s.ai_predicted_class, s.ai_confidence, p.aadhar_number
        FROM scans s
        JOIN patients p ON s.aadhar_number = p.aadhar_number
        WHERE p.o_id = ? AND s.doctor_verified_class IS NULL
        ORDER BY s.scan_date DESC
    """, (o_id,))
    unverified = cursor.fetchall()
    conn.close()
    return unverified

def submit_doctor_review(scan_id, verified_class):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT ai_predicted_class FROM scans WHERE scan_id=?", (scan_id,))
    ai_class = cursor.fetchone()[0]
    is_correct = (int(verified_class) == int(ai_class))
    
    cursor.execute("""
        UPDATE scans 
        SET doctor_verified_class=?, is_ai_correct=?, ready_for_retraining=1
        WHERE scan_id=?
    """, (verified_class, is_correct, scan_id))
    conn.commit()
    conn.close()

# --- AI MODEL ---
@st.cache_resource
def load_model():
    return DRAgent(model_path=os.path.join(MODELS_DIR, "dr_model_weights.pth"))

try:
    agent = load_model()
except Exception as e:
    st.error(f"Error loading AI: {e}")
    st.stop()

# --- SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['doctor_name'] = ""
    st.session_state['doctor_id'] = None

# ==========================================
# UI: AUTHENTICATION
# ==========================================
if not st.session_state['logged_in']:
    st.title("👁️ Clinic Portal Login")
    tab1, tab2 = st.tabs(["Login", "Register New Clinic"])
    
    with tab1:
        login_email = st.text_input("Email", key="log_email")
        login_pass = st.text_input("Password", type="password", key="log_pass")
        if st.button("Login"):
            user = login_doctor(login_email, login_pass)
            if user:
                st.session_state['logged_in'] = True
                st.session_state['doctor_id'] = user[0]
                st.session_state['doctor_name'] = user[1]
                st.rerun()
            else:
                st.error("Invalid email or password.")
                
    with tab2:
        reg_name = st.text_input("Full Name (Dr.)")
        reg_email = st.text_input("Email ID")
        reg_pass = st.text_input("Password", type="password")
        reg_dob = st.date_input("Date of Birth")
        reg_address = st.text_area("Clinic Address")
        if st.button("Register"):
            if register_doctor(reg_name, reg_email, reg_pass, reg_dob, reg_address):
                st.success("Registration successful! Please login.")
            else:
                st.error("Email already registered.")

# ==========================================
# UI: MAIN DASHBOARD
# ==========================================
else:
    st.sidebar.title(f"Welcome, Dr. {st.session_state['doctor_name']}")
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['doctor_id'] = None # Ensure deep logout
        st.rerun()
        
    st.title("Diabetic Retinopathy Diagnostic Agent")
    dash_tab, history_tab, mgmt_tab = st.tabs(["New Patient Scan", "Patient History", "Patient Management"])
    
    # --- TAB 1: NEW SCAN ---
    with dash_tab:
        st.markdown("### Upload New Scan")
        patient_name = st.text_input("Enter Patient Name:")
        aadhar_number = st.text_input("Enter Aadhar Number (12 digits):", max_chars=12)
        uploaded_file = st.file_uploader("Drop patient eye scan here (.jpg or .png)", type=["jpg", "jpeg", "png"])

        if uploaded_file and patient_name and aadhar_number and len(aadhar_number) == 12 and aadhar_number.isdigit():
            if st.button("Analyze & Save to Patient Record"):
                col1, col2 = st.columns(2)
                
                timestamp = int(time.time())
                perm_scan_path = os.path.join(SCANS_DIR, f"scan_{timestamp}.png")
                perm_heatmap_path = os.path.join(HEATMAPS_DIR, f"heatmap_{timestamp}.jpg")
                
                with open(perm_scan_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with col1:
                    st.subheader("Original Scan")
                    # BUG FIX 1: Updated to use_container_width
                    st.image(perm_scan_path, use_container_width=True)

                with col2:
                    st.subheader("AI Analysis")
                    with st.spinner('Neural Network analyzing...'):
                        results = agent.run_analysis(perm_scan_path)
                        temp_heatmap = results["Heatmap_Saved_At"]
                        shutil.move(temp_heatmap, perm_heatmap_path)
                        
                        save_clinical_record(
                            st.session_state['doctor_id'], 
                            patient_name, 
                            aadhar_number,
                            perm_scan_path, 
                            perm_heatmap_path, 
                            results['DR_Class'], 
                            results['Confidence']
                        )
                        
                    # BUG FIX 1: Updated to use_container_width
                    st.image(perm_heatmap_path, use_container_width=True)
                    st.metric(label="Predicted Diagnosis", value=f"Class {results['DR_Class']}")
                    st.metric(label="AI Confidence", value=results['Confidence'])
                    
                    st.info(results['Explanation'])
                    
                    if "Struggling" in results['Agent_Struggles']:
                        st.warning(results['Agent_Struggles'])
                    else:
                        st.success(results['Agent_Struggles'])
                        
                    st.success("✅ Record successfully saved to Clinical Database.")

    # --- TAB 2: HISTORY ---
    with history_tab:
        st.markdown("### Patient History")
        
        history = get_doctor_history(st.session_state['doctor_id'])
        
        if not history:
            st.info("No patient records found. Analyze a scan to build history.")
        else:
            for record in history:
                scan_id, p_name, date, orig_img, heat_img, ai_class, ai_conf, doc_class, aadhar = record
                
                # Reconstruct absolute paths safely from DB records
                orig_img_path = os.path.join(SCANS_DIR, os.path.basename(orig_img.replace('\\', '/')))
                heat_img_path = os.path.join(HEATMAPS_DIR, os.path.basename(heat_img.replace('\\', '/')))

                with st.expander(f"Patient: {p_name} (Aadhar: {aadhar}) | Date: {date[:10]}"):
                    h_col1, h_col2 = st.columns(2)
                    with h_col1:
                        st.image(orig_img_path, caption="Original Scan", use_container_width=True)
                    with h_col2:
                        st.image(heat_img_path, caption=f"AI Heatmap (Confidence: {ai_conf*100:.1f}%)", use_container_width=True)
                    
                    st.divider()
                    
                    st.metric(label="AI Predicted Diagnosis", value=f"Class {ai_class}")
                    
                    if doc_class is None:
                        st.warning("Needs Doctor Verification (Go to Patient Management Tab)")
                    else:
                        st.success(f"✅ Verified by Ophthalmologist as: Class {doc_class}")
                        if doc_class == ai_class:
                            st.write("AI was **Correct**.")
                        else:
                            st.write(f"AI was **Incorrect**. Saved to retraining pipeline.")

    # --- TAB 3: PATIENT MANAGEMENT ---
    with mgmt_tab:
        st.markdown("### Patient Management & Verification")
        
        unverified_scans = get_unverified_scans(st.session_state['doctor_id'])
        
        if not unverified_scans:
            st.success("All scans have been verified! No pending reviews.")
        else:
            for record in unverified_scans:
                scan_id, p_name, date, orig_img, heat_img, ai_class, ai_conf, aadhar = record
                
                orig_img_path = os.path.join(SCANS_DIR, os.path.basename(orig_img.replace('\\', '/')))
                heat_img_path = os.path.join(HEATMAPS_DIR, os.path.basename(heat_img.replace('\\', '/')))

                with st.expander(f"Review Needed: {p_name} (Aadhar: {aadhar}) | Date: {date[:10]}", expanded=True):
                    h_col1, h_col2 = st.columns(2)
                    with h_col1:
                        st.image(orig_img_path, caption="Original Scan", use_container_width=True)
                    with h_col2:
                        st.image(heat_img_path, caption=f"AI Heatmap (Confidence: {ai_conf*100:.1f}%)", use_container_width=True)
                    
                    st.divider()
                    st.write(f"**AI Predicted Diagnosis:** Class {ai_class}")
                    
                    with st.form(key=f"form_verify_{scan_id}"):
                        true_label = st.selectbox("Select True Clinical Diagnosis:", [0, 1, 2, 3, 4], index=ai_class)
                        if st.form_submit_button("Confirm & Add to Retraining Data"):
                            submit_doctor_review(scan_id, true_label)
                            st.success("Feedback recorded! Refreshing...")
                            time.sleep(1)
                            st.rerun()