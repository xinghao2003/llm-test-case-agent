# Login Web Application

A minimal Flask web application that provides a secure email + password sign-in flow. It validates credential input, redirects authenticated users to a dashboard, displays clear error messages, masks passwords in the form, and logs failed sign-in attempts for audit purposes.

## Features mapped to the acceptance criteria

- Rejects empty email or password fields with a clear validation message.
- Confirms that the submitted email exists in the seeded user data set.
- Verifies passwords by comparing SHA-256 hashes (no plain-text storage).
- Redirects authenticated users to `/dashboard` after a successful login.
- Shows the exact error message `Invalid email or password` for incorrect credentials.
- Keeps the password field masked via the `type="password"` input.
- Records failed login attempts (email and client IP) to `login_attempts.log` for security auditing.

## Getting started

1. **Install dependencies**

   ```powershell
   pip install flask
   ```

2. **Run the app**

   ```powershell
   cd examples/login_app
   python app.py
   ```

3. **Visit the login page**

   Open <http://127.0.0.1:5000/> in your browser.

4. **Sign in with the seeded account**

   - Email: `user@example.com`
   - Password: `SecurePass123!`

   Successful sign-in leads to the dashboard; failed attempts log to `login_attempts.log`.

## Customization notes

- To add or update users, edit `data/users.json` and store SHA-256 password hashes. You can generate a hash in Python with:

  ```python
  import hashlib
  hashlib.sha256("your-new-password".encode()).hexdigest()
  ```

- Override the Flask secret key in production by setting the `LOGIN_APP_SECRET` environment variable.

## Project structure

```text
login_app/
├── app.py                 # Flask routes and authentication helpers
├── data/users.json        # Seeded user records with hashed passwords
├── login_attempts.log     # Created on-demand to capture failed sign-ins
├── static/styles.css      # Simple styling for the pages
└── templates/
   ├── dashboard.html     # Post-login landing page
   └── login.html         # Sign-in form with validation messaging
```

## Configuration

- Set `LOGIN_APP_SECRET` to override the default Flask session secret in production.
- Update `data/users.json` with additional accounts, ensuring each `password_hash` is a SHA-256 hash.
- Adjust logging behavior by editing the handler configuration near the top of `app.py`.

## Security considerations

- Passwords are never stored in plain text; hashes are computed with SHA-256.
- Failed login attempts are timestamped and recorded, making it easier to monitor suspicious activity.
- The form enforces required fields client-side and server-side; empty submissions are rejected before attempting authentication.
