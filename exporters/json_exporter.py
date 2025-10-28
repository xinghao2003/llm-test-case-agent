"""
Export test cases as JSON
"""
import json
import re
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class JSONExporter:
    """
    Exports test cases in structured JSON format
    """

    def __init__(self):
        """Initialize the JSON exporter"""
        pass

    def export(
        self,
        tests: str,
        output_path: str,
        metadata: Dict = None
    ) -> str:
        """
        Export tests to JSON format

        Args:
            tests: Test code string
            output_path: Path to save the JSON file
            metadata: Additional metadata

        Returns:
            Path to the exported file
        """
        # Parse tests into structured format
        structured_tests = self._parse_tests(tests)

        # Build JSON structure
        json_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'test_count': len(structured_tests),
                'format_version': '1.0'
            },
            'tests': structured_tests
        }

        # Add additional metadata if provided
        if metadata:
            json_data['metadata'].update(metadata)

        # Ensure .json extension
        output_path = Path(output_path)
        if output_path.suffix != '.json':
            output_path = output_path.with_suffix('.json')

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(json_data, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )

        return str(output_path)

    def _parse_tests(self, tests: str) -> List[Dict]:
        """
        Parse test code into structured format

        Args:
            tests: Test code string

        Returns:
            List of test dictionaries
        """
        structured_tests = []

        # Split into individual test functions
        test_pattern = r'def (test_\w+)\([^)]*\):(.*?)(?=\ndef |\Z)'
        matches = re.findall(test_pattern, tests, re.DOTALL)

        for test_name, test_body in matches:
            # Extract docstring
            docstring_match = re.search(r'"""(.*?)"""', test_body, re.DOTALL)
            docstring = docstring_match.group(1).strip() if docstring_match else ""

            # Extract assertions
            assertions = re.findall(r'assert .+', test_body)

            # Extract arrange/act/assert comments
            sections = {
                'arrange': self._extract_section(test_body, 'Arrange'),
                'act': self._extract_section(test_body, 'Act'),
                'assert': self._extract_section(test_body, 'Assert')
            }

            structured_test = {
                'name': test_name,
                'description': docstring,
                'assertions': assertions,
                'assertion_count': len(assertions),
                'sections': sections,
                'code': f'def {test_name}():{test_body}'
            }

            # Detect test type
            if 'pytest.raises' in test_body or 'with pytest.raises' in test_body:
                structured_test['type'] = 'exception_test'
            elif 'fixture' in test_body.lower():
                structured_test['type'] = 'fixture_test'
            else:
                structured_test['type'] = 'standard_test'

            structured_tests.append(structured_test)

        return structured_tests

    def _extract_section(self, test_body: str, section_name: str) -> str:
        """
        Extract Arrange/Act/Assert section from test body

        Args:
            test_body: Test function body
            section_name: Section to extract (Arrange, Act, or Assert)

        Returns:
            Section code or empty string
        """
        pattern = rf'# {section_name}\s*\n(.*?)(?=# |$)'
        match = re.search(pattern, test_body, re.DOTALL | re.IGNORECASE)

        if match:
            return match.group(1).strip()

        return ""
