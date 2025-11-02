import json
import logging
import os
from hashlib import sha256
from pathlib import Path
from typing import Dict, Optional

from flask import Flask, redirect, render_template, request, session, url_for, flash

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "users.json"
LOG_PATH = BASE_DIR / "login_attempts.log"

app = Flask(__name__)
app.secret_key = os.environ.get("LOGIN_APP_SECRET", "dev-secret-key-change-me")
app.config.update(SESSION_COOKIE_HTTPONLY=True)

logger = logging.getLogger("login_app")
logger.setLevel(logging.INFO)
if not logger.handlers:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)


def load_users() -> Dict[str, Dict[str, str]]:
    with DATA_PATH.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    return {entry["email"].lower(): entry for entry in records if "email" in entry and "password_hash" in entry}


USERS = load_users()


def hash_password(raw_password: str) -> str:
    return sha256(raw_password.encode("utf-8")).hexdigest()


def authenticate(email: str, password: str) -> Optional[Dict[str, str]]:
    user = USERS.get(email.lower())
    if not user:
        return None
    if user["password_hash"] != hash_password(password):
        return None
    return user


@app.route("/", methods=["GET", "POST"])
def login():
    email = ""
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Email and password are required.", "error")
            return render_template("login.html", email=email)

        user = authenticate(email, password)
        if not user:
            logger.info("Failed login attempt - email=%s ip=%s", email, request.remote_addr or "unknown")
            flash("Invalid email or password", "error")
            return render_template("login.html", email=email)

        session["user_email"] = user["email"]
        session["display_name"] = user.get("display_name", user["email"])
        return redirect(url_for("dashboard"))

    return render_template("login.html", email=email)


@app.route("/dashboard")
def dashboard():
    if "user_email" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", display_name=session.get("display_name", session["user_email"]))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
