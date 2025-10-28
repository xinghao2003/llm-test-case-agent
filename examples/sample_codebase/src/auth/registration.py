"""
User registration and authentication functions
"""
import re
import hashlib
from typing import Optional
from src.models.user import User
from src.auth.exceptions import (
    DuplicateEmailError,
    InvalidCredentialsError,
    InactiveAccountError,
    WeakPasswordError,
    InvalidEmailError
)


# Simulated in-memory database
_users_db = {}
_user_id_counter = 1


def register_user(email: str, password: str) -> User:
    """
    Register a new user account

    Args:
        email: User's email address
        password: Plain text password

    Returns:
        Created User instance

    Raises:
        DuplicateEmailError: If email already exists
        InvalidEmailError: If email format is invalid
        WeakPasswordError: If password doesn't meet requirements
    """
    global _user_id_counter

    # Validate email
    if not _is_valid_email(email):
        raise InvalidEmailError(f"Invalid email format: {email}")

    # Check for duplicate
    if email.lower() in _users_db:
        raise DuplicateEmailError(f"Email already registered: {email}")

    # Validate password strength
    if not _is_strong_password(password):
        raise WeakPasswordError(
            "Password must be at least 8 characters with numbers and special characters"
        )

    # Hash password
    password_hash = _hash_password(password)

    # Create user
    user = User(
        user_id=_user_id_counter,
        email=email.lower(),
        password_hash=password_hash
    )

    # Store in database
    _users_db[email.lower()] = user
    _user_id_counter += 1

    return user


def authenticate_user(email: str, password: str) -> User:
    """
    Authenticate a user with email and password

    Args:
        email: User's email address
        password: Plain text password

    Returns:
        Authenticated User instance

    Raises:
        InvalidCredentialsError: If email/password combination is invalid
        InactiveAccountError: If account is deactivated
    """
    # Find user
    user = _users_db.get(email.lower())

    if not user:
        raise InvalidCredentialsError("Invalid email or password")

    # Check password
    password_hash = _hash_password(password)
    if user.password_hash != password_hash:
        raise InvalidCredentialsError("Invalid email or password")

    # Check if account is active
    if not user.is_active:
        raise InactiveAccountError("Account is deactivated")

    # Update last login
    user.update_last_login()

    return user


def get_user_by_email(email: str) -> Optional[User]:
    """
    Retrieve a user by email address

    Args:
        email: Email address to search for

    Returns:
        User instance if found, None otherwise
    """
    return _users_db.get(email.lower())


def delete_user(email: str) -> bool:
    """
    Delete a user account

    Args:
        email: Email of user to delete

    Returns:
        True if user was deleted, False if not found
    """
    email = email.lower()
    if email in _users_db:
        del _users_db[email]
        return True
    return False


def _is_valid_email(email: str) -> bool:
    """
    Validate email format

    Args:
        email: Email to validate

    Returns:
        True if valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def _is_strong_password(password: str) -> bool:
    """
    Check if password meets strength requirements

    Requirements:
    - At least 8 characters
    - Contains at least one number
    - Contains at least one special character

    Args:
        password: Password to check

    Returns:
        True if strong, False otherwise
    """
    if len(password) < 8:
        return False

    has_number = any(c.isdigit() for c in password)
    has_special = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password)

    return has_number and has_special


def _hash_password(password: str) -> str:
    """
    Hash a password using SHA-256

    Args:
        password: Plain text password

    Returns:
        Hashed password
    """
    return hashlib.sha256(password.encode()).hexdigest()


def reset_database():
    """Reset the in-memory database (for testing)"""
    global _users_db, _user_id_counter
    _users_db = {}
    _user_id_counter = 1
