# backend/server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
import uuid
import time
import numpy as np
import hashlib
from datetime import datetime

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
FL_DIR = os.path.join(BASE_DIR, 'fl_server')
STORAGE_DIR = os.path.join(FL_DIR, 'storage')
USERS_FILE = os.path.join(STORAGE_DIR, 'users.json')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
CHAT_FILE = os.path.join(BASE_DIR, 'chat_messages.json')  # chat storage

os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ==========================
# USER STORAGE
# ==========================
USERS = {}
OTP_STORE = {}

def _load_users_file():
    global USERS
    if not os.path.exists(USERS_FILE):
        USERS = {}
        return
    try:
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # If file contains {"users": [...]} convert list to map
        if isinstance(data, dict) and 'users' in data and isinstance(data['users'], list):
            mapping = {}
            for entry in data['users']:
                if isinstance(entry, dict) and 'username' in entry:
                    mapping[entry['username']] = entry
            USERS = mapping
        elif isinstance(data, dict):
            USERS = data
        else:
            USERS = {}
    except Exception:
        USERS = {}

def save_users():
    tmp = USERS_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(USERS, f, indent=2)
        os.replace(tmp, USERS_FILE)
    except Exception:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(USERS, f, indent=2)

def make_token():
    return str(uuid.uuid4())

def verify_password(user, pwd):
    if user.get("password_hash"):
        try:
            return check_password_hash(user["password_hash"], pwd)
        except Exception:
            return False
    if user.get("password"):
        return hashlib.sha256(pwd.encode()).hexdigest() == user["password"]
    return False

# Load users on server start
_load_users_file()

# ==========================
# REGISTER
# ==========================
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip()
    password = data.get("password") or ""

    if not username or not password or not email:
        return jsonify({"error": "username, email, password required"}), 400

    if username in USERS:
        return jsonify({"error": "username_taken"}), 400

    USERS[username] = {
        "username": username,
        "email": email,
        "password_hash": generate_password_hash(password),
        "created_at": datetime.utcnow().isoformat(),
        "token": None
    }
    save_users()
    return jsonify({"status": "ok", "username": username})

# ==========================
# LOGIN
# ==========================
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    u = USERS.get(username)
    if not u or not verify_password(u, password):
        return jsonify({"error": "invalid_credentials"}), 401

    token = make_token()
    u["token"] = token
    u["last_login"] = datetime.utcnow().isoformat()
    USERS[username] = u
    save_users()

    return jsonify({
        "status": "ok",
        "token": token,
        "user": {"username": username, "email": u.get("email")}
    })

# ==========================
# OTP FORGOT PASSWORD
# ==========================
@app.route("/request_otp", methods=["POST"])
def request_otp():
    data = request.get_json() or {}
    username = (data.get("username") or "").strip()

    if username not in USERS:
        return jsonify({"error": "unknown_user"}), 400

    otp = str(100000 + (uuid.uuid4().int % 900000))
    OTP_STORE[username] = {"otp": otp, "ts": time.time()}

    print(f"[DEBUG OTP] {username} -> {otp}")
    return jsonify({"status": "ok", "debug_otp": otp})

@app.route("/verify_otp", methods=["POST"])
def verify_otp():
    data = request.get_json() or {}
    username = (data.get("username") or "").strip()
    otp = data.get("otp")
    new_pwd = data.get("new_password")

    rec = OTP_STORE.get(username)
    if not rec or rec.get("otp") != otp:
        return jsonify({"error": "invalid_otp"}), 400

    USERS[username]["password_hash"] = generate_password_hash(new_pwd)
    save_users()
    OTP_STORE.pop(username, None)
    return jsonify({"status": "ok"})

# ==========================
# PREDICT (DEMO) - saves a report file
# ==========================
def _make_report_file(username, report_obj):
    # create a unique id for report
    rid = str(uuid.uuid4())
    timestamp = int(time.time() * 1000)
    record = {
        "id": rid,
        "username": username,
        "report": report_obj,
        "timestamp": timestamp,
        "status": "pending",       # pending by default for doctor review
        "doctor_notes": ""         # empty until doctor updates
    }
    fname = os.path.join(REPORTS_DIR, f"report_{username}_{rid}.json")
    tmp = fname + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, fname)
    return record

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    img_b64 = data.get("image_b64")
    username = (data.get("username") or "guest").strip()

    resp = {
        "label": "No Tumor",
        "confidence": 0.93,
        "model_version": "v1",
        "notes": "Demo prediction — replace with real model inference"
    }

    # Save richer report file (structured for later doctor review)
    try:
        record = _make_report_file(username, resp)
        print(f"[REPORT SAVED] {record['id']} for {username}")
    except Exception as e:
        print("Failed saving report:", e)

    return jsonify(resp)

# ==========================
# CHAT ENDPOINTS
# ==========================
def _ensure_chat_file():
    if not os.path.exists(CHAT_FILE):
        try:
            with open(CHAT_FILE, "w", encoding="utf-8") as fh:
                json.dump([], fh)
        except Exception:
            pass

@app.route("/send_message", methods=["POST"])
def send_message():
    data = request.get_json() or {}
    sender = (data.get("sender") or "").strip()
    receiver = (data.get("receiver") or "").strip()
    message = data.get("message", "")
    timestamp = int(time.time())

    if not sender or not receiver or message is None:
        return jsonify({"error": "missing_fields"}), 400

    _ensure_chat_file()

    try:
        with open(CHAT_FILE, "r", encoding="utf-8") as fh:
            messages = json.load(fh)
            if not isinstance(messages, list):
                messages = []
    except Exception:
        messages = []

    entry = {
        "id": str(uuid.uuid4()),
        "sender": sender,
        "receiver": receiver,
        "message": message,
        "timestamp": timestamp
    }

    messages.append(entry)

    try:
        tmp = CHAT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(messages, fh, indent=2)
        os.replace(tmp, CHAT_FILE)
    except Exception:
        # fallback write
        with open(CHAT_FILE, "w", encoding="utf-8") as fh:
            json.dump(messages, fh, indent=2)

    return jsonify({"success": True, "message_id": entry["id"]})

@app.route("/get_messages", methods=["POST"])
def get_messages():
    data = request.get_json() or {}
    user1 = (data.get("user1") or "").strip()
    user2 = (data.get("user2") or "").strip()

    if not user1 or not user2:
        return jsonify({"messages": []})

    _ensure_chat_file()

    try:
        with open(CHAT_FILE, "r", encoding="utf-8") as fh:
            messages = json.load(fh)
            if not isinstance(messages, list):
                messages = []
    except Exception:
        messages = []

    filtered = [
        msg for msg in messages
        if (msg.get("sender") == user1 and msg.get("receiver") == user2)
        or (msg.get("sender") == user2 and msg.get("receiver") == user1)
    ]

    # sort by timestamp ascending
    filtered.sort(key=lambda x: x.get("timestamp", 0))
    return jsonify({"messages": filtered})

# ==========================
# FEDERATED LEARNING ENDPOINTS
# ==========================
@app.route("/upload_weights", methods=["POST"])
def upload_weights():
    data = request.get_json() or {}
    username = data.get("username")
    weights = data.get("weights")

    if not username or weights is None:
        return jsonify({"error": "missing"}), 400

    try:
        arr = np.array(weights, dtype=np.float32)
        fname = os.path.join(STORAGE_DIR, f"{username}_weights_{int(time.time())}.npz")
        np.savez_compressed(fname, arr=arr)
        return jsonify({"status": "ok", "file": os.path.basename(fname)})
    except Exception as e:
        return jsonify({"error": "save_failed", "message": str(e)}), 500

@app.route("/get_global_meta", methods=["GET"])
def get_global_meta():
    return jsonify({
        "version": "v1",
        "updated": time.time()
    })

@app.route("/trigger_aggregation", methods=["POST"])
def trigger_aggregation():
    # simple stub that pretends aggregation started
    print("[FL] Aggregation triggered by admin")
    return jsonify({"status": "ok", "message": "aggregation_started"})

# ==========================
# REPORTS FOR PATIENT & DOCTOR
# ==========================
@app.route("/list_reports", methods=["POST"])
def list_reports():
    """
    Request body: { "username": "<username or 'all' or 'pending'>" }
    - username == "all"  -> return every report (doctor)
    - username == "pending" -> return only pending reports (doctor)
    - otherwise return only reports for that username (patient)
    """
    data = request.get_json() or {}
    username = (data.get("username") or "").strip()

    files = os.listdir(REPORTS_DIR)
    results = []

    for f in files:
        path = os.path.join(REPORTS_DIR, f)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                j = json.load(fh)
            if username == "all":
                results.append(j)
            elif username == "pending":
                if j.get("status") == "pending":
                    results.append(j)
            else:
                if j.get("username") == username:
                    results.append(j)
        except Exception:
            # ignore malformed files
            continue

    # return list under 'reports' key (Flutter expects this)
    return jsonify({"reports": results})

# ==========================
# DOCTOR: update a report (approve/reject/save notes)
# ==========================
@app.route("/doctor_update_report", methods=["POST"])
def doctor_update_report():
    """
    Expected payload:
    {
      "report_id": "<id>",
      "doctor_notes": "<text>",
      "action": "approve"|"reject"|"save"
    }
    """
    data = request.get_json() or {}
    report_id = (data.get("report_id") or "").strip()
    doctor_notes = data.get("doctor_notes", "")
    action = (data.get("action") or "save").strip().lower()

    if not report_id:
        return jsonify({"error": "report_id required"}), 400

    # find file by id
    files = os.listdir(REPORTS_DIR)
    updated = False
    for f in files:
        path = os.path.join(REPORTS_DIR, f)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                j = json.load(fh)
            if j.get("id") == report_id:
                # update fields
                j["doctor_notes"] = doctor_notes
                if action == "approve":
                    j["status"] = "approved"
                elif action == "reject":
                    j["status"] = "rejected"
                else:
                    # save only
                    j["status"] = j.get("status", "pending")
                # write back atomically
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as out:
                    json.dump(j, out, indent=2)
                os.replace(tmp, path)
                updated = True
                break
        except Exception:
            continue

    if not updated:
        return jsonify({"error": "not_found"}), 404

    return jsonify({"status": "ok", "message": "report_updated"})

# ==========================
# RUN SERVER
# ==========================
if __name__ == "__main__":
    print("✔ Backend running at http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
