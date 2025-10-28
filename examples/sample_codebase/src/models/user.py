"""
User model for authentication system
"""
from datetime import datetime
from typing import Optional


class User:
    """
    Represents a user in the system
    """

    def __init__(
        self,
        user_id: int,
        email: str,
        password_hash: str,
        created_at: Optional[datetime] = None
    ):
        """
        Initialize a User instance

        Args:
            user_id: Unique user identifier
            email: User's email address
            password_hash: Hashed password
            created_at: Account creation timestamp
        """
        self.user_id = user_id
        self.email = email
        self.password_hash = password_hash
        self.created_at = created_at or datetime.now()
        self.is_active = True
        self.last_login = None

    def __repr__(self):
        return f"<User(id={self.user_id}, email={self.email})>"

    def to_dict(self) -> dict:
        """
        Convert user to dictionary representation

        Returns:
            Dictionary with user data
        """
        return {
            'user_id': self.user_id,
            'email': self.email,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }

    def update_last_login(self):
        """Update the last login timestamp to now"""
        self.last_login = datetime.now()

    def deactivate(self):
        """Deactivate the user account"""
        self.is_active = False

    def activate(self):
        """Activate the user account"""
        self.is_active = True
