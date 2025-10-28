"""
Authentication exceptions
"""


class AuthenticationError(Exception):
    """Base exception for authentication errors"""
    pass


class DuplicateEmailError(AuthenticationError):
    """Raised when attempting to register with an existing email"""
    pass


class InvalidCredentialsError(AuthenticationError):
    """Raised when login credentials are invalid"""
    pass


class InactiveAccountError(AuthenticationError):
    """Raised when attempting to login with an inactive account"""
    pass


class WeakPasswordError(AuthenticationError):
    """Raised when password doesn't meet strength requirements"""
    pass


class InvalidEmailError(AuthenticationError):
    """Raised when email format is invalid"""
    pass
