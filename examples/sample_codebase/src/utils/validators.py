"""
Email validation utilities
"""
import re
from typing import List


def validate_email_format(email: str) -> bool:
    """
    Validate email address format

    Args:
        email: Email address to validate

    Returns:
        True if valid format, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def extract_domain(email: str) -> str:
    """
    Extract domain from email address

    Args:
        email: Email address

    Returns:
        Domain portion of email

    Raises:
        ValueError: If email format is invalid
    """
    if '@' not in email:
        raise ValueError("Invalid email format")

    return email.split('@')[1]


def is_corporate_email(email: str, corporate_domains: List[str]) -> bool:
    """
    Check if email is from a corporate domain

    Args:
        email: Email address to check
        corporate_domains: List of corporate domains

    Returns:
        True if email is from corporate domain
    """
    try:
        domain = extract_domain(email)
        return domain.lower() in [d.lower() for d in corporate_domains]
    except ValueError:
        return False


def sanitize_email(email: str) -> str:
    """
    Sanitize email address (lowercase and strip whitespace)

    Args:
        email: Email address to sanitize

    Returns:
        Sanitized email address
    """
    return email.strip().lower()
