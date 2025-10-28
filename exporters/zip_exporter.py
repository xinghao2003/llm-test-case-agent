"""
Export codebase with integrated test files as ZIP
"""
import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class ZipCodebaseExporter:
    """
    Exports a complete codebase with integrated test files as ZIP
    """

    def __init__(self):
        """Initialize the ZIP exporter"""
        pass

    def export(
        self,
        original_codebase_path: str,
        tests: str,
        test_file_path: str,
        output_path: str,
        metadata: Dict = None
    ) -> str:
        """
        Export codebase with integrated tests as ZIP

        Args:
            original_codebase_path: Path to original codebase
            tests: Generated test code
            test_file_path: Where to place test file (relative to project root)
            output_path: Output ZIP file path
            metadata: Additional metadata

        Returns:
            Path to exported ZIP file
        """
        # Ensure .zip extension
        output_path = Path(output_path)
        if output_path.suffix != '.zip':
            output_path = output_path.with_suffix('.zip')

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create temporary directory
        temp_dir = output_path.parent / f"temp_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        temp_dir.mkdir(exist_ok=True)

        try:
            # Copy original codebase
            if Path(original_codebase_path).exists():
                self._copy_codebase(original_codebase_path, temp_dir)

            # Write test file
            test_file_full_path = temp_dir / test_file_path
            test_file_full_path.parent.mkdir(parents=True, exist_ok=True)

            # Add header to test file
            test_content = self._generate_test_file_header(metadata) + "\n\n" + tests

            test_file_full_path.write_text(test_content, encoding='utf-8')

            # Create __init__.py in test directory if needed
            test_init = test_file_full_path.parent / '__init__.py'
            if not test_init.exists():
                test_init.write_text('"""Test package"""', encoding='utf-8')

            # Create README for tests
            readme_path = temp_dir / 'TEST_README.md'
            readme_path.write_text(self._generate_test_readme(metadata), encoding='utf-8')

            # Update requirements.txt if exists
            self._update_requirements(temp_dir)

            # Create ZIP file
            self._create_zip(temp_dir, output_path)

            return str(output_path)

        finally:
            # Cleanup temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def _copy_codebase(self, source: str, destination: Path):
        """
        Copy codebase, excluding unnecessary files

        Args:
            source: Source directory
            destination: Destination directory
        """
        source = Path(source)

        # Directories to exclude
        exclude_dirs = {
            '__pycache__', '.git', '.venv', 'venv', 'env',
            'node_modules', '.pytest_cache', '.mypy_cache',
            'htmlcov', 'dist', 'build', '.tox', 'eggs',
            '.eggs', '*.egg-info', '.coverage'
        }

        # File patterns to exclude
        exclude_files = {
            '*.pyc', '*.pyo', '*.pyd', '.DS_Store', '*.so',
            '*.dylib', '*.db', '*.sqlite'
        }

        def ignore_patterns(dir, files):
            """Pattern for shutil.copytree ignore"""
            ignored = []
            for item in files:
                # Ignore directories
                if item in exclude_dirs:
                    ignored.append(item)
                # Ignore file patterns
                for pattern in exclude_files:
                    if pattern.startswith('*.') and item.endswith(pattern[1:]):
                        ignored.append(item)
                        break
                    elif item == pattern:
                        ignored.append(item)
                        break
            return ignored

        shutil.copytree(source, destination, ignore=ignore_patterns, dirs_exist_ok=True)

    def _generate_test_file_header(self, metadata: Dict = None) -> str:
        """Generate header comment for test file"""
        lines = [
            '"""',
            'AI-Generated Test Suite',
            '=' * 60,
            ''
        ]

        if metadata:
            lines.append('Generation Information:')
            if 'user_story' in metadata:
                lines.append(f"User Story: {metadata['user_story']}")
            if 'timestamp' in metadata:
                lines.append(f"Generated: {metadata['timestamp']}")
            if 'coverage' in metadata:
                lines.append(f"Coverage: {metadata['coverage']:.1%}")
            if 'iterations' in metadata:
                lines.append(f"Iterations: {metadata['iterations']}")
            lines.append('')

        lines.extend([
            'IMPORTANT:',
            '- Review these tests before running them',
            '- Adjust imports if your project structure differs',
            '- Add fixtures as needed for your specific setup',
            '- Some tests may need manual adjustment for your environment',
            '',
            'To run:',
            '  pytest ' + (metadata.get('test_file_path', 'tests/') if metadata else 'tests/'),
            '',
            'Generated by AI-Powered Test Case Generator',
            'https://github.com/yourusername/joyce-fyp',
            '"""'
        ])

        return '\n'.join(lines)

    def _generate_test_readme(self, metadata: Dict = None) -> str:
        """Generate README for running tests"""
        content = """# Generated Test Suite

This codebase has been augmented with AI-generated test cases.

## Test Location

Generated tests are located in: `{test_location}`

## Running Tests

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov={module} --cov-report=html
```

### Run Specific Test File

```bash
pytest {test_file}
```

## Test Information

{metadata_info}

## Important Notes

1. **Review Before Running**: These tests were AI-generated based on user stories. Please review them before running in production.

2. **Adjust Imports**: If your project structure is different, you may need to adjust import statements.

3. **Add Fixtures**: Some tests may require pytest fixtures specific to your setup (database connections, mock objects, etc.).

4. **Configuration**: Update `pytest.ini` or `setup.cfg` if needed for your project.

5. **Environment**: Ensure your test environment is properly configured (env variables, test database, etc.).

## Test Coverage Areas

The generated tests aim to cover:
- Happy path scenarios
- Edge cases and boundary values
- Error handling and exceptions
- Input validation
- Security considerations (where applicable)

## Feedback

If tests need adjustment or improvement, you can:
1. Manually edit the test files
2. Re-generate with more specific user story context
3. Add custom fixtures and helpers

## Generated By

AI-Powered Test Case Generator
- Project: https://github.com/yourusername/joyce-fyp
- Documentation: See README.md in the generator project

---

**Happy Testing!** 🧪
"""
        # Format with actual values
        test_location = metadata.get('test_file_path', 'tests/') if metadata else 'tests/'
        test_file = metadata.get('test_file_path', 'tests/test_generated.py') if metadata else 'tests/test_generated.py'

        module = 'your_module'  # Could extract from project structure
        if metadata and 'project_path' in metadata:
            project_path = Path(metadata['project_path'])
            module = project_path.name

        # Metadata info
        metadata_info = ""
        if metadata:
            info_lines = []
            if 'user_story' in metadata:
                info_lines.append(f"- **User Story**: {metadata['user_story']}")
            if 'coverage' in metadata:
                info_lines.append(f"- **Coverage**: {metadata['coverage']:.1%}")
            if 'iterations' in metadata:
                info_lines.append(f"- **Iterations**: {metadata['iterations']}")
            if 'test_count' in metadata:
                info_lines.append(f"- **Test Count**: {metadata['test_count']}")
            if 'timestamp' in metadata:
                info_lines.append(f"- **Generated**: {metadata['timestamp']}")

            metadata_info = '\n'.join(info_lines)

        return content.format(
            test_location=test_location,
            test_file=test_file,
            module=module,
            metadata_info=metadata_info
        )

    def _update_requirements(self, project_dir: Path):
        """
        Update or create requirements.txt with test dependencies

        Args:
            project_dir: Project directory
        """
        req_file = project_dir / 'requirements.txt'

        # Test dependencies to add
        test_deps = [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0'
        ]

        existing_deps = set()

        # Read existing requirements
        if req_file.exists():
            existing_content = req_file.read_text()
            existing_deps = set(existing_content.strip().split('\n'))

        # Add test dependencies if not present
        for dep in test_deps:
            dep_name = dep.split('>=')[0].split('==')[0]
            if not any(dep_name in line for line in existing_deps):
                existing_deps.add(dep)

        # Write updated requirements
        req_file.write_text('\n'.join(sorted(existing_deps)) + '\n', encoding='utf-8')

    def _create_zip(self, source_dir: Path, output_path: Path):
        """
        Create ZIP file from directory

        Args:
            source_dir: Directory to zip
            output_path: Output ZIP file path
        """
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(source_dir)
                    zipf.write(file_path, arcname)

    def export_tests_only(
        self,
        tests: str,
        output_path: str,
        metadata: Dict = None
    ) -> str:
        """
        Export only the test file (not full codebase)

        Args:
            tests: Test code
            output_path: Output path
            metadata: Metadata

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add header
        content = self._generate_test_file_header(metadata) + "\n\n" + tests

        output_path.write_text(content, encoding='utf-8')

        return str(output_path)
