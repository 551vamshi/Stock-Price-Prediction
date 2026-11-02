"""
Simple local authentication for the Stock Price Prediction Streamlit app.

Accounts are stored in users.json beside this file.
Passwords are stored as PBKDF2-HMAC-SHA256 hashes, never as plain text.
"""

import hashlib
import hmac
import json
import secrets
from pathlib import Path

import streamlit as st

USERS_FILE = Path(__file__).resolve().parent / "users.json"
PBKDF2_ITERATIONS = 310_000


def _load_users():
    if not USERS_FILE.exists():
        return {}
    try:
        with USERS_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        st.error("The local users.json file could not be read. Check the file and try again.")
        return {}


def _save_users(users):
    temporary_file = USERS_FILE.with_suffix(".tmp")
    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(users, file, indent=2)
    temporary_file.replace(USERS_FILE)


def _normalise_username(value):
    return value.strip().lower()


def _hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    ).hex()


def _register_user(name, username, email, password):
    users = _load_users()
    username_key = _normalise_username(username)
    email_key = email.strip().lower()

    if not name.strip() or not username_key or not email_key:
        return False, "Please complete all fields."
    if len(username_key) < 3:
        return False, "Username must contain at least 3 characters."
    if len(password) < 8:
        return False, "Password must contain at least 8 characters."
    if username_key in users:
        return False, "That username is already registered."

    if any(user.get("email", "").lower() == email_key for user in users.values()):
        return False, "That email address is already registered."

    salt = secrets.token_bytes(16)
    users[username_key] = {
        "name": name.strip(),
        "email": email_key,
        "salt": salt.hex(),
        "password_hash": _hash_password(password, salt),
    }
    _save_users(users)
    return True, "Account created. You can now log in."


def _verify_user(username_or_email, password):
    users = _load_users()
    entered = username_or_email.strip().lower()

    for username_key, user in users.items():
        if entered in (username_key, user.get("email", "").lower()):
            try:
                salt = bytes.fromhex(user["salt"])
                expected_hash = user["password_hash"]
            except (KeyError, ValueError, TypeError):
                return None

            actual_hash = _hash_password(password, salt)
            if hmac.compare_digest(actual_hash, expected_hash):
                return {
                    "username": username_key,
                    "name": user.get("name", username_key),
                    "email": user.get("email", ""),
                }
            return None

    return None


def require_auth():
    """Show login/signup until authenticated; return the current user."""
    if "authenticated_user" in st.session_state:
        return st.session_state.authenticated_user

    st.markdown(
        """
        <style>
        .auth-wrap {max-width: 520px; margin: 2rem auto 0 auto;}
        .auth-heading {text-align:center; margin-bottom: .25rem;}
        .auth-subtitle {text-align:center; color:#9ca3af; margin-bottom:1.5rem;}
        </style>
        <div class="auth-wrap">
          <h1 class="auth-heading">📈 Stock Price Prediction</h1>
          <p class="auth-subtitle">Deep Learning &amp; Data Analytics Dashboard</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    login_tab, signup_tab = st.tabs(["🔐 Login", "📝 Sign Up"])

    with login_tab:
        with st.form("login_form"):
            login_id = st.text_input("Username or Email")
            login_password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button(
                "Login", type="primary", use_container_width=True
            )

        if login_submitted:
            if not login_id.strip() or not login_password:
                st.error("Enter your username/email and password.")
            else:
                user = _verify_user(login_id, login_password)
                if user:
                    st.session_state.authenticated_user = user
                    st.success("Login successful.")
                    st.rerun()
                else:
                    st.error("Incorrect username/email or password.")

    with signup_tab:
        with st.form("signup_form"):
            name = st.text_input("Full Name")
            new_username = st.text_input("Create Username")
            email = st.text_input("Email Address")
            new_password = st.text_input("Create Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            signup_submitted = st.form_submit_button(
                "Create Account", type="primary", use_container_width=True
            )

        if signup_submitted:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = _register_user(
                    name, new_username, email, new_password
                )
                if success:
                    st.success(message)
                    st.info("Open the Login tab and sign in with your new account.")
                else:
                    st.error(message)

    st.caption("Educational mini-project authentication. Accounts are stored locally on this computer.")
    st.stop()
