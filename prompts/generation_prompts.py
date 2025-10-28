"""
Test generation prompt templates
"""

INITIAL_GENERATION_PROMPT = """You are an expert software testing engineer specializing in Python and pytest. Your task is to generate comprehensive pytest test functions for the following user story.

USER STORY:
{user_story}

CONTEXT:
{additional_context}

REQUIREMENTS:
- Generate valid Python pytest functions
- Include clear docstrings explaining each test
- Use appropriate pytest fixtures and markers where applicable
- Include assertions with descriptive failure messages
- Follow pytest naming conventions (test_*)
- Consider: happy path, edge cases, error conditions, boundary values
- Add comments for test setup (Arrange), execution (Act), and verification (Assert)

OUTPUT FORMAT:
Return ONLY valid Python code with pytest test functions. Do not include explanations outside the code.

Example structure:
```python
import pytest

def test_happy_path_scenario():
    \"\"\"Test the main success path\"\"\"
    # Arrange
    input_data = "valid_input"

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result == expected_output, "Should return expected output for valid input"

def test_edge_case_empty_input():
    \"\"\"Test behavior with empty input\"\"\"
    # Arrange
    input_data = ""

    # Act & Assert
    with pytest.raises(ValueError, match="Input cannot be empty"):
        function_under_test(input_data)
```

Generate comprehensive tests covering:
1. Happy path (main success scenario)
2. Edge cases (boundary values, empty inputs, null values)
3. Error conditions (invalid inputs, exceptions)
4. Input validation scenarios

GENERATED TESTS:
"""

ITERATIVE_GENERATION_PROMPT = """You are an expert software testing engineer performing iterative test improvement. You have already generated some tests, and now you need to generate ADDITIONAL tests to fill coverage gaps.

USER STORY:
{user_story}

EXISTING TESTS:
{existing_tests}

COVERAGE GAPS IDENTIFIED:
{gaps}

FOCUS AREAS:
{focus_areas}

REQUIREMENTS:
- Generate ONLY NEW tests that aren't already covered in the existing tests
- Address the specific gaps mentioned above
- Maintain the same code quality and style as existing tests
- Follow pytest best practices
- Include descriptive docstrings

OUTPUT FORMAT:
Return ONLY valid Python code with NEW pytest test functions. Do not repeat existing tests.

ADDITIONAL TESTS:
"""

REFINEMENT_PROMPT = """You are refining existing pytest test functions based on user feedback and clarifications.

ORIGINAL TESTS:
{original_tests}

USER FEEDBACK:
{user_feedback}

CLARIFICATIONS PROVIDED:
{clarifications}

REQUIREMENTS:
- Update the tests based on the feedback and clarifications
- Maintain pytest compatibility
- Preserve good tests that don't need changes
- Improve tests that need refinement
- Add any missing tests based on new information

Return the COMPLETE updated test suite with all improvements applied.

REFINED TESTS:
"""

SECURITY_FOCUSED_PROMPT = """You are a security-focused testing expert. Generate pytest tests specifically for security scenarios.

USER STORY:
{user_story}

EXISTING TESTS:
{existing_tests}

Generate additional tests focusing on:
1. Input validation and sanitization
2. SQL injection prevention
3. XSS (Cross-Site Scripting) prevention
4. Authentication and authorization
5. Rate limiting
6. Data encryption/protection
7. Session management
8. CSRF protection (if applicable)

Return ONLY new security-focused pytest test functions.

SECURITY TESTS:
"""

CODEBASE_AWARE_GENERATION_PROMPT = """You are an expert software testing engineer specializing in Python and pytest. Generate pytest tests for a user story by analyzing the actual codebase.

USER STORY:
{user_story}

CONTEXT:
{additional_context}

CODEBASE INFORMATION:
{codebase_context}

REQUIREMENTS:
- Import actual functions/classes from the codebase using the correct module paths
- Use real function signatures (parameters, return types)
- Generate tests that will actually work with the provided code
- Include appropriate imports at the top
- Follow the existing code style and patterns
- Use real data types and structures from the codebase

IMPORTANT:
- Use "from {module_path} import {function_name}" for actual imports
- Call functions with correct parameter names and types
- Assert on actual return types and structures
- Consider the actual implementation's edge cases

Example with real imports:
```python
import pytest
from src.auth.registration import register_user, UserAlreadyExistsError
from src.models.user import User

def test_successful_registration():
    \"\"\"Test user registration with valid credentials\"\"\"
    # Arrange
    email = "test@example.com"
    password = "SecurePass123!"

    # Act
    result = register_user(email, password)

    # Assert
    assert isinstance(result, User)
    assert result.email == email
    assert result.password != password  # Should be hashed

def test_duplicate_registration_raises_error():
    \"\"\"Test that duplicate email raises UserAlreadyExistsError\"\"\"
    # Arrange
    email = "existing@example.com"
    password = "Pass123!"
    register_user(email, password)  # First registration

    # Act & Assert
    with pytest.raises(UserAlreadyExistsError):
        register_user(email, password)  # Duplicate attempt
```

Generate comprehensive tests covering:
1. Happy path scenarios with actual function calls
2. Edge cases based on actual code logic
3. Error handling with actual exception types
4. Input validation using actual validators
5. Integration points with real dependencies

GENERATED TESTS:
"""

CODEBASE_ITERATIVE_PROMPT = """Generate ADDITIONAL tests for existing codebase tests, filling coverage gaps.

USER STORY:
{user_story}

EXISTING TESTS:
{existing_tests}

CODEBASE INFORMATION:
{codebase_context}

COVERAGE GAPS:
{gaps}

FOCUS AREAS:
{focus_areas}

Generate ONLY NEW tests that:
- Use actual imports from the codebase
- Address the identified gaps
- Don't duplicate existing tests
- Call real functions with correct signatures

Return ONLY valid Python code with NEW pytest test functions.

ADDITIONAL TESTS:
"""
