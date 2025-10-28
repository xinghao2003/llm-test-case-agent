# Codebase Analysis Mode - Usage Guide

## Overview

The AI Test Case Generator now supports **two modes**:

1. **User Story Only Mode** (Original): Generate test templates from requirements
2. **Codebase Analysis Mode** (NEW): Generate tests that work with your actual code

## Codebase Analysis Mode

### What It Does

Instead of generating placeholder tests, the system:
1. Analyzes your actual Python code
2. Finds functions and classes relevant to your user story
3. Generates tests with real imports and function calls
4. Creates tests that can actually run against your codebase

### How to Use

#### Basic Python API Usage

```python
from parsers import CodebaseAnalyzer
from agents import TestGenerator

# 1. Analyze your codebase
analyzer = CodebaseAnalyzer()
relevant_code = analyzer.find_relevant_code(
    user_story="As a user, I want to register an account",
    project_path="/path/to/your/project"
)

# 2. Format for prompt
codebase_context = analyzer.format_code_context_for_prompt(relevant_code)

# 3. Generate tests with codebase awareness
generator = TestGenerator()
result = generator.generate_from_codebase(
    user_story="As a user, I want to register an account",
    codebase_context=codebase_context,
    additional_context="Email and password required"
)

print(result['code'])
```

### Example: Before vs After

#### Before (User Story Only)

**Input:**
```
User Story: "As a user, I want to register with email and password"
```

**Output:**
```python
import pytest

def test_successful_registration():
    email = "test@example.com"
    password = "pass123"
    result = register_user(email, password)  # ← Placeholder!
    assert result['success'] is True
```

**Problem**: `register_user` doesn't exist, won't run!

#### After (With Codebase Analysis)

**Input:**
```
User Story: "As a user, I want to register with email and password"
Codebase: /my/project
```

**System analyzes and finds:**
```
File: src/auth/registration.py
Function: def create_user(email: str, pwd: str) -> User
Class: User (in src/models/user.py)
Exception: DuplicateEmailError
```

**Output:**
```python
import pytest
from src.auth.registration import create_user, DuplicateEmailError
from src.models.user import User

def test_successful_registration():
    """Test user registration with valid credentials"""
    # Arrange
    email = "test@example.com"
    password = "SecurePass123!"

    # Act
    result = create_user(email, password)  # ← Real function!

    # Assert
    assert isinstance(result, User)
    assert result.email == email
    assert result.is_active is True

def test_duplicate_email_raises_error():
    """Test that duplicate email raises DuplicateEmailError"""
    # Arrange
    email = "existing@example.com"
    create_user(email, "pass1")

    # Act & Assert
    with pytest.raises(DuplicateEmailError):  # ← Real exception!
        create_user(email, "pass2")
```

**Result**: Tests import real code, call real functions, can actually run!

## Export Options

### 1. Documentation Only (PDF, CSV, JSON)

Export test specifications without codebase:

```python
from exporters import PDFExporter

exporter = PDFExporter()
exporter.export(
    tests=generated_tests,
    output_path="test_spec.pdf",
    metadata={'user_story': story, 'coverage': 0.85}
)
```

**Use Case**: Share test plans with team, documentation

### 2. Integrated Codebase ZIP

Export your codebase WITH tests integrated:

```python
from exporters import ZipCodebaseExporter

exporter = ZipCodebaseExporter()
exporter.export(
    original_codebase_path="/my/project",
    tests=generated_tests,
    test_file_path="tests/test_registration.py",
    output_path="project_with_tests.zip",
    metadata={'user_story': story, 'coverage': 0.85}
)
```

**What's included:**
- Your entire codebase (excluding .git, venv, etc.)
- Generated test file in correct location
- Updated requirements.txt with pytest
- TEST_README.md with instructions
- `__init__.py` in test directory

**Result**: Complete, ready-to-run project!

## When to Use Each Mode

### Use User Story Only Mode When:
- ✅ You're in the design phase (code doesn't exist yet)
- ✅ You want test specifications/documentation
- ✅ You're defining requirements before coding
- ✅ You need test templates to fill in later
- ✅ Working with non-Python projects (just for specs)

### Use Codebase Analysis Mode When:
- ✅ You have existing Python code to test
- ✅ You want tests that actually run
- ✅ You're doing regression testing
- ✅ You need to add tests to legacy code
- ✅ You want realistic integration tests

## Supported Features

### Code Analysis
- ✅ Function signature extraction
- ✅ Class and method discovery
- ✅ Type annotation parsing
- ✅ Docstring extraction
- ✅ Import path generation
- ✅ Relevance scoring (matches user story keywords)

### Test Generation
- ✅ Real import statements
- ✅ Correct function calls with parameters
- ✅ Actual exception types
- ✅ Type-aware assertions
- ✅ Integration with existing code patterns

### Export
- ✅ ZIP with full codebase + tests
- ✅ Proper directory structure
- ✅ Updated requirements.txt
- ✅ Test runner instructions
- ✅ README for tests

## Limitations

### Current Version (PoC)
- Python only (other languages in future)
- Analyzes up to 50 files (configurable)
- Simple keyword-based relevance matching
- Doesn't execute code (static analysis only)
- Doesn't analyze external dependencies

### Planned Enhancements
- Semantic code search (embeddings)
- Support for JavaScript/TypeScript
- Dependency graph analysis
- Mock generation for external APIs
- Test data inference from code
- Fixture generation

## API Reference

### CodebaseAnalyzer

```python
class CodebaseAnalyzer:
    def analyze_project(project_path: str) -> Dict:
        """Analyze entire project structure"""

    def find_relevant_code(user_story: str, project_path: str) -> Dict:
        """Find code relevant to user story"""

    def format_code_context_for_prompt(relevant_code: Dict) -> str:
        """Format for LLM prompt"""
```

### TestGenerator (New Methods)

```python
class TestGenerator:
    def generate_from_codebase(
        user_story: str,
        codebase_context: str,
        additional_context: str = ""
    ) -> Dict:
        """Generate tests with codebase awareness"""

    def generate_additional_tests_from_codebase(
        user_story: str,
        existing_tests: str,
        codebase_context: str,
        gaps: List[Dict],
        focus_areas: List[str] = None
    ) -> Dict:
        """Iterative generation with codebase"""
```

### ZipCodebaseExporter

```python
class ZipCodebaseExporter:
    def export(
        original_codebase_path: str,
        tests: str,
        test_file_path: str,
        output_path: str,
        metadata: Dict = None
    ) -> str:
        """Export codebase with integrated tests"""

    def export_tests_only(
        tests: str,
        output_path: str,
        metadata: Dict = None
    ) -> str:
        """Export just the test file"""
```

## Example Workflow

### Complete End-to-End Example

```python
#!/usr/bin/env python3
"""
Example: Generate tests for existing Flask app
"""
from parsers import CodebaseAnalyzer
from agents import TestGenerator, TestValidator
from exporters import ZipCodebaseExporter

# Step 1: Define user story
user_story = """
As an API user, I want to create a new user account via POST /api/users
with email and password, receiving a 201 status and user ID on success.
"""

# Step 2: Analyze codebase
analyzer = CodebaseAnalyzer()
relevant_code = analyzer.find_relevant_code(
    user_story=user_story,
    project_path="./my_flask_app"
)

print(f"Found {len(relevant_code['relevant_functions'])} relevant functions")
print(f"Found {len(relevant_code['relevant_classes'])} relevant classes")

# Step 3: Generate tests
generator = TestGenerator(api_key="your_key")
codebase_context = analyzer.format_code_context_for_prompt(relevant_code)

result = generator.generate_from_codebase(
    user_story=user_story,
    codebase_context=codebase_context,
    additional_context="Flask app with SQLAlchemy"
)

# Step 4: Validate
validator = TestValidator()
validation = validator.evaluate(result['code'])

if validation['syntax_valid'] and validation['pytest_compatible']:
    print(f"✅ Generated {validation['test_count']} valid tests")
else:
    print(f"⚠️ Issues: {validation['quality_issues']}")

# Step 5: Export
exporter = ZipCodebaseExporter()
zip_path = exporter.export(
    original_codebase_path="./my_flask_app",
    tests=result['code'],
    test_file_path="tests/test_api_users.py",
    output_path="./my_flask_app_with_tests.zip",
    metadata={
        'user_story': user_story,
        'test_count': validation['test_count'],
        'coverage': 0.85
    }
)

print(f"📦 Exported to: {zip_path}")
print("Next steps:")
print("  1. Unzip the file")
print("  2. cd into the directory")
print("  3. pip install -r requirements.txt")
print("  4. pytest tests/test_api_users.py")
```

## Troubleshooting

### "No relevant code found"

**Problem**: Analyzer can't match user story to code

**Solutions**:
- Add more keywords to user story
- Specify file/module names in context
- Use action verbs (create, update, delete, etc.)
- Mention specific entities (User, Order, etc.)

### "Generated tests won't run"

**Problem**: Import errors or function not found

**Solutions**:
- Check project_path is correct
- Ensure code files are in standard locations
- Verify Python path structure matches
- Manually adjust imports if needed

### "Tests generated but not finding functions"

**Problem**: Code structure doesn't match expectations

**Solutions**:
- Provide actual function names in context
- Check if functions are in `__init__.py`
- Verify module structure
- Add type hints to help analyzer

## Best Practices

1. **Start Specific**: Use precise user stories with entity/action names
2. **Provide Context**: Add function names, file paths in additional context
3. **Review First**: Always review generated tests before running
4. **Iterate**: Use the iterative mode to fill gaps
5. **Manual Tuning**: Adjust imports/fixtures as needed for your setup

## Future Roadmap

- [ ] JavaScript/TypeScript support
- [ ] Semantic code search
- [ ] Automatic fixture generation
- [ ] Mock object creation
- [ ] Test data inference
- [ ] CI/CD integration
- [ ] GitHub Actions workflow
- [ ] VSCode extension

---

**Ready to try it?** See example usage in `/examples/codebase_analysis_example.py`
