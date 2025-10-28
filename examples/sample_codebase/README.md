# Sample Authentication System

A simple Python authentication system used as an example for demonstrating the AI Test Case Generator's codebase analysis feature.

## Structure

```
src/
├── __init__.py
├── models/
│   └── user.py          # User model class
├── auth/
│   ├── registration.py  # Registration and authentication
│   └── exceptions.py    # Custom exceptions
└── utils/
    └── validators.py    # Email validation utilities
```

## Features

- User registration with email/password
- Email format validation
- Password strength requirements
- User authentication
- Account activation/deactivation
- In-memory database (for demo purposes)

## Example Usage

```python
from src.auth.registration import register_user, authenticate_user
from src.auth.exceptions import DuplicateEmailError, InvalidCredentialsError

# Register a new user
user = register_user("john@example.com", "SecurePass123!")
print(f"Created user: {user.email}")

# Authenticate
authenticated_user = authenticate_user("john@example.com", "SecurePass123!")
print(f"Logged in: {authenticated_user.email}")
```

## Use with Test Generator

This codebase is designed to demonstrate the AI Test Case Generator's ability to:

1. Analyze real Python code
2. Extract functions, classes, and signatures
3. Generate tests with actual imports
4. Create runnable test suites

See `examples/codebase_analysis_example.py` for how to generate tests for this code.
