#!/usr/bin/env python3
"""
Complete Example: Codebase Analysis Mode

This script demonstrates how to use the AI Test Case Generator's
codebase analysis feature to generate runnable tests for actual Python code.

Example Codebase: Sample authentication system in ./sample_codebase/
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers import CodebaseAnalyzer
from agents import TestGenerator
from exporters import ZipCodebaseExporter, PythonExporter


def main():
    """Run the complete codebase analysis example"""

    print("=" * 70)
    print("AI Test Case Generator - Codebase Analysis Example")
    print("=" * 70)
    print()

    # Configuration
    example_dir = Path(__file__).parent
    codebase_path = example_dir / "sample_codebase"

    if not codebase_path.exists():
        print(f"❌ Error: Example codebase not found at {codebase_path}")
        print("Make sure you're running this from the examples/ directory")
        return

    # Step 1: Define User Story
    print("📝 STEP 1: User Story")
    print("-" * 70)

    user_story = """
    As a user, I want to register an account with email and password
    so that I can access the system.
    """

    additional_context = """
    - Email must be valid format (user@domain.com)
    - Password must be at least 8 characters
    - Password must contain numbers and special characters
    - System should prevent duplicate email registrations
    - Account can be activated or deactivated
    """

    print(f"User Story: {user_story.strip()}")
    print(f"\nContext: {additional_context.strip()}")
    print()

    # Step 2: Analyze Codebase
    print("🔍 STEP 2: Analyze Codebase")
    print("-" * 70)

    analyzer = CodebaseAnalyzer()

    print(f"Analyzing: {codebase_path}")
    print("Searching for relevant code...")

    relevant_code = analyzer.find_relevant_code(
        user_story=user_story,
        project_path=str(codebase_path)
    )

    print(f"\n✅ Analysis Complete!")
    print(f"   - Found {len(relevant_code['relevant_files'])} relevant files")
    print(f"   - Found {len(relevant_code['relevant_functions'])} relevant functions")
    print(f"   - Found {len(relevant_code['relevant_classes'])} relevant classes")
    print()

    # Display top relevant functions
    print("📌 Top Relevant Functions:")
    for i, func in enumerate(relevant_code['relevant_functions'][:5], 1):
        print(f"   {i}. {func['name']}() - {func['file']}")
        print(f"      Signature: {func['signature']}")
        if func.get('docstring'):
            doc_preview = func['docstring'].split('\n')[0][:60]
            print(f"      Doc: {doc_preview}...")
        print()

    # Display relevant classes
    if relevant_code['relevant_classes']:
        print("📌 Relevant Classes:")
        for i, cls in enumerate(relevant_code['relevant_classes'][:3], 1):
            print(f"   {i}. {cls['name']} - {cls['file']}")
            if cls.get('methods'):
                print(f"      Methods: {', '.join([m['name'] for m in cls['methods'][:3]])}")
        print()

    # Step 3: Format for LLM
    print("📄 STEP 3: Format Context for LLM")
    print("-" * 70)

    codebase_context = analyzer.format_code_context_for_prompt(relevant_code)
    print("Context formatted for AI prompt:")
    print(codebase_context[:500] + "...\n")

    # Step 4: Generate Tests (requires API key)
    print("🤖 STEP 4: Generate Tests with Codebase Awareness")
    print("-" * 70)

    api_key = os.getenv('OPENROUTER_API_KEY')

    if not api_key:
        print("⚠️  OPENROUTER_API_KEY not set")
        print("   To generate actual tests, set your API key:")
        print("   export OPENROUTER_API_KEY='your_key_here'")
        print()
        print("📋 For now, showing what WOULD be generated...")
        print()
        show_expected_output()
        return

    print("Generating tests with real imports and function calls...")
    print("(This may take 30-60 seconds)")
    print()

    try:
        generator = TestGenerator(api_key=api_key)

        result = generator.generate_from_codebase(
            user_story=user_story,
            codebase_context=codebase_context,
            additional_context=additional_context
        )

        print("✅ Test Generation Complete!")
        print(f"   - Generated {result['metadata']['test_count']} test functions")
        print(f"   - Has real imports: {result['metadata']['has_real_imports']}")
        print()

        # Display generated tests
        print("📝 Generated Test Code:")
        print("=" * 70)
        print(result['code'])
        print("=" * 70)
        print()

        # Step 5: Export Options
        print("📤 STEP 5: Export Options")
        print("-" * 70)

        # Option 1: Python file only
        python_exporter = PythonExporter()
        py_path = python_exporter.export(
            result['code'],
            str(example_dir / "generated_tests.py"),
            metadata={
                'user_story': user_story,
                'test_count': result['metadata']['test_count']
            }
        )
        print(f"✅ Exported Python file: {py_path}")

        # Option 2: ZIP with full codebase
        print("\n📦 Exporting complete project with integrated tests...")

        zip_exporter = ZipCodebaseExporter()
        zip_path = zip_exporter.export(
            original_codebase_path=str(codebase_path),
            tests=result['code'],
            test_file_path="tests/test_user_registration.py",
            output_path=str(example_dir / "sample_codebase_with_tests.zip"),
            metadata={
                'user_story': user_story.strip(),
                'test_count': result['metadata']['test_count'],
                'coverage': 0.85,
                'project_path': str(codebase_path)
            }
        )

        print(f"✅ Exported ZIP: {zip_path}")
        print()
        print("📋 Next Steps:")
        print(f"   1. unzip {Path(zip_path).name}")
        print("   2. cd sample_codebase_with_tests")
        print("   3. pip install pytest")
        print("   4. pytest tests/test_user_registration.py")
        print()

    except Exception as e:
        print(f"❌ Error during generation: {str(e)}")
        print()
        print("📋 Showing expected output instead...")
        print()
        show_expected_output()


def show_expected_output():
    """Show what the expected generated tests would look like"""

    print("=" * 70)
    print("EXPECTED OUTPUT (Example of Generated Tests)")
    print("=" * 70)
    print()

    example_tests = '''import pytest
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


@pytest.fixture(autouse=True)
def reset_test_database():
    """Reset database before each test"""
    reset_database()
    yield
    reset_database()


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
    assert user.user_id is not None


def test_duplicate_email_raises_error():
    """Test that registering with duplicate email raises DuplicateEmailError"""
    # Arrange
    email = "jane@example.com"
    password = "ValidPass456!"
    register_user(email, password)  # First registration

    # Act & Assert
    with pytest.raises(DuplicateEmailError, match="Email already registered"):
        register_user(email, password)  # Duplicate attempt


def test_invalid_email_format_raises_error():
    """Test that invalid email format raises InvalidEmailError"""
    # Arrange
    invalid_emails = ["notanemail", "missing@domain", "@nodomain.com"]
    password = "ValidPass123!"

    # Act & Assert
    for email in invalid_emails:
        with pytest.raises(InvalidEmailError, match="Invalid email format"):
            register_user(email, password)


def test_weak_password_raises_error():
    """Test that weak password raises WeakPasswordError"""
    # Arrange
    email = "user@example.com"
    weak_passwords = ["short", "noNumbers!", "NoSpecialChars123"]

    # Act & Assert
    for password in weak_passwords:
        with pytest.raises(WeakPasswordError):
            register_user(email, password)


def test_successful_authentication():
    """Test user can authenticate with correct credentials"""
    # Arrange
    email = "auth@example.com"
    password = "SecurePass789!"
    register_user(email, password)

    # Act
    authenticated_user = authenticate_user(email, password)

    # Assert
    assert isinstance(authenticated_user, User)
    assert authenticated_user.email == email.lower()
    assert authenticated_user.last_login is not None


def test_authentication_with_wrong_password_raises_error():
    """Test authentication fails with incorrect password"""
    # Arrange
    email = "user@example.com"
    correct_password = "RightPass123!"
    wrong_password = "WrongPass456!"
    register_user(email, correct_password)

    # Act & Assert
    with pytest.raises(InvalidCredentialsError, match="Invalid email or password"):
        authenticate_user(email, wrong_password)


def test_authentication_with_nonexistent_email_raises_error():
    """Test authentication fails with non-existent email"""
    # Arrange
    email = "nonexistent@example.com"
    password = "AnyPass123!"

    # Act & Assert
    with pytest.raises(InvalidCredentialsError, match="Invalid email or password"):
        authenticate_user(email, password)


def test_deactivated_account_cannot_login():
    """Test that deactivated account cannot authenticate"""
    # Arrange
    email = "inactive@example.com"
    password = "ValidPass123!"
    user = register_user(email, password)
    user.deactivate()

    # Act & Assert
    with pytest.raises(InactiveAccountError, match="Account is deactivated"):
        authenticate_user(email, password)


def test_get_user_by_email_returns_user():
    """Test retrieving user by email"""
    # Arrange
    email = "retrieve@example.com"
    password = "ValidPass123!"
    register_user(email, password)

    # Act
    user = get_user_by_email(email)

    # Assert
    assert user is not None
    assert user.email == email.lower()


def test_get_user_by_email_returns_none_for_nonexistent():
    """Test retrieving non-existent user returns None"""
    # Act
    user = get_user_by_email("nonexistent@example.com")

    # Assert
    assert user is None


def test_delete_user_removes_account():
    """Test that delete_user removes user from system"""
    # Arrange
    email = "delete@example.com"
    password = "ValidPass123!"
    register_user(email, password)

    # Act
    result = delete_user(email)

    # Assert
    assert result is True
    assert get_user_by_email(email) is None


def test_password_is_hashed_not_plain_text():
    """Test that passwords are stored as hashes, not plain text"""
    # Arrange
    email = "secure@example.com"
    password = "PlainTextPassword123!"

    # Act
    user = register_user(email, password)

    # Assert
    assert user.password_hash != password
    assert len(user.password_hash) > len(password)  # Hash is longer
'''

    print(example_tests)
    print()
    print("=" * 70)
    print("Key Features in Generated Tests:")
    print("=" * 70)
    print("✅ Real imports from actual codebase modules")
    print("✅ Uses actual exception classes (DuplicateEmailError, etc.)")
    print("✅ Calls real functions (register_user, authenticate_user)")
    print("✅ References actual User model class")
    print("✅ Includes pytest fixtures for test isolation")
    print("✅ Tests happy path, edge cases, and error scenarios")
    print("✅ Ready to run with: pytest test_user_registration.py")
    print()


if __name__ == "__main__":
    main()
