# Examples Directory

This directory contains complete examples demonstrating both modes of the AI Test Case Generator.

## 📂 Contents

### 1. Sample Codebase (`sample_codebase/`)

A realistic Python authentication system used to demonstrate **codebase analysis mode**.

**Structure:**
```
sample_codebase/
├── src/
│   ├── models/
│   │   └── user.py          # User model class
│   ├── auth/
│   │   ├── registration.py  # Registration & authentication
│   │   └── exceptions.py    # Custom exception classes
│   └── utils/
│       └── validators.py    # Email validation
└── README.md
```

**Features:**
- User registration with email/password validation
- Password strength requirements
- Email format validation
- Account activation/deactivation
- Custom exception classes
- Complete type hints

### 2. Codebase Analysis Example (`codebase_analysis_example.py`)

Complete working example showing the **full workflow** of codebase analysis mode.

**What it does:**
1. ✅ Analyzes the sample codebase
2. ✅ Finds relevant functions and classes
3. ✅ Formats context for LLM
4. ✅ Generates tests with real imports
5. ✅ Exports as Python file and ZIP

**Run it:**
```bash
cd examples
python codebase_analysis_example.py
```

### 3. Generated Test Examples

#### `example_generated_tests.py`
Example of tests generated in **user story only mode** (templates with placeholders).

#### Output from `codebase_analysis_example.py`
When run, generates:
- `generated_tests.py` - Tests with real imports
- `sample_codebase_with_tests.zip` - Complete project ready to run

### 4. User Story Example (`example_user_registration.md`)

Detailed user story with context showing what to include for best results.

---

## 🚀 Quick Start

### Try Codebase Analysis Mode

```bash
# 1. Set your API key
export OPENROUTER_API_KEY='your_key_here'

# 2. Run the example
cd examples
python codebase_analysis_example.py

# 3. Check the output
ls -l generated_tests.py
ls -l sample_codebase_with_tests.zip

# 4. Test the generated tests
unzip sample_codebase_with_tests.zip
cd sample_codebase_with_tests
pip install pytest
pytest tests/test_user_registration.py
```

**Without API Key:**

The example script will still run and show you:
- What code it found
- What context was extracted
- Expected output (example tests)

---

## 📋 Example Output

### Step 1: Analysis Results

```
🔍 STEP 2: Analyze Codebase
----------------------------------------------------------------------
Analyzing: ./sample_codebase
Searching for relevant code...

✅ Analysis Complete!
   - Found 3 relevant files
   - Found 5 relevant functions
   - Found 1 relevant classes

📌 Top Relevant Functions:
   1. register_user() - .../src/auth/registration.py
      Signature: def register_user(email: str, password: str) -> User
      Doc: Register a new user account...

   2. authenticate_user() - .../src/auth/registration.py
      Signature: def authenticate_user(email: str, password: str) -> User
      Doc: Authenticate a user with email and password...
```

### Step 2: Generated Tests (with real imports!)

```python
import pytest
from src.auth.registration import (
    register_user,
    authenticate_user,
    get_user_by_email,
    delete_user,
    reset_database
)
from src.models.user import User
from src.auth.exceptions import (
    DuplicateEmailError,
    InvalidCredentialsError,
    InactiveAccountError,
    WeakPasswordError,
    InvalidEmailError
)


def test_successful_user_registration():
    """Test user registration with valid email and password"""
    # Arrange
    email = "john@example.com"
    password = "SecurePass123!"

    # Act
    user = register_user(email, password)

    # Assert
    assert isinstance(user, User)
    assert user.email == email.lower()
    assert user.is_active is True
```

**Notice:**
- ✅ Real imports from `src.auth.registration`
- ✅ Actual exception classes
- ✅ Real User model
- ✅ Calls actual functions
- ✅ Ready to run!

---

## 🎯 Comparison: User Story vs Codebase Mode

### User Story Only Mode

**Input:**
```
User story: "As a user, I want to register with email and password"
```

**Output:**
```python
import pytest

def test_user_registration():
    email = "test@example.com"
    password = "pass123"
    result = register_user(email, password)  # ← Placeholder!
    assert result.success is True
```

**Can't run** - function doesn't exist!

### Codebase Analysis Mode

**Input:**
```
User story: "As a user, I want to register with email and password"
Codebase: ./sample_codebase
```

**Output:**
```python
import pytest
from src.auth.registration import register_user  # ← Real import!
from src.models.user import User
from src.auth.exceptions import DuplicateEmailError

def test_user_registration():
    user = register_user("test@example.com", "SecurePass123!")
    assert isinstance(user, User)
    assert user.email == "test@example.com"

def test_duplicate_email_raises_error():
    register_user("test@example.com", "pass123")
    with pytest.raises(DuplicateEmailError):  # ← Real exception!
        register_user("test@example.com", "pass456")
```

**Can run!** - Real imports, real functions!

---

## 📚 Learning Path

### Beginner: Understand the Example

1. Read `sample_codebase/README.md`
2. Browse the code in `sample_codebase/src/`
3. Look at `example_user_registration.md`

### Intermediate: Run the Example

1. Set your API key
2. Run `python codebase_analysis_example.py`
3. Examine the generated tests
4. Run the tests with pytest

### Advanced: Adapt for Your Project

1. Study the example script
2. Replace `sample_codebase` with your project path
3. Customize the user story
4. Generate tests for your code!

---

## 🔧 Using with Your Own Code

### Quick Template

```python
from parsers import CodebaseAnalyzer
from agents import TestGenerator
from exporters import ZipCodebaseExporter

# 1. Analyze your codebase
analyzer = CodebaseAnalyzer()
code = analyzer.find_relevant_code(
    user_story="Your user story here",
    project_path="/path/to/your/project"
)

# 2. Generate tests
generator = TestGenerator(api_key="your_key")
context = analyzer.format_code_context_for_prompt(code)
tests = generator.generate_from_codebase(
    user_story="Your user story here",
    codebase_context=context
)

# 3. Export
ZipCodebaseExporter().export(
    original_codebase_path="/path/to/your/project",
    tests=tests['code'],
    test_file_path="tests/test_your_feature.py",
    output_path="your_project_with_tests.zip"
)
```

---

## 💡 Tips for Best Results

### Write Good User Stories

**Bad:**
```
"Test the user stuff"
```

**Good:**
```
As a user, I want to register an account with email and password
so that I can access the system securely.

Context:
- Email must be valid format
- Password minimum 8 characters with numbers and special characters
- Prevent duplicate registrations
- Account can be activated/deactivated
```

### Project Structure

Works best with:
- ✅ Standard Python package structure
- ✅ Clear module organization
- ✅ Type hints on functions
- ✅ Docstrings explaining purpose
- ✅ Descriptive function/class names

### Keywords Matter

Include relevant keywords in your user story:
- Action verbs: register, login, create, update, delete
- Entities: user, account, email, password
- Requirements: validation, authentication, authorization

---

## 🐛 Troubleshooting

### "No relevant code found"

**Problem:** Analyzer can't match story to code

**Solutions:**
- Add more specific keywords to user story
- Mention actual function/class names in context
- Use action verbs that appear in your code

### "Generated tests won't run"

**Problem:** Import errors or missing modules

**Solutions:**
- Check `project_path` is correct
- Ensure Python can find your modules
- Verify package structure (needs `__init__.py`)
- Manually adjust imports if needed

### "Not finding my functions"

**Problem:** Code structure doesn't match expectations

**Solutions:**
- Ensure functions are at module level (not nested)
- Check file extensions are `.py`
- Verify files aren't in excluded directories
- Add type hints to help analysis

---

## 📖 Further Reading

- `../README.md` - Main project documentation
- `../CODEBASE_MODE_USAGE.md` - Complete codebase mode guide
- `../ARCHITECTURE.md` - System design details
- `../QUICKSTART.md` - 5-minute quick start

---

## 🎓 Next Steps

1. **Run the example** - See it in action
2. **Examine the output** - Understand what it generates
3. **Try with your code** - Apply to your project
4. **Iterate and improve** - Refine based on results

---

**Happy Testing!** 🧪✨
