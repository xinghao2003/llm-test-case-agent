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
