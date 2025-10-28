"""
Export test cases as CSV (Test Case Matrix)
"""
import csv
import re
from pathlib import Path
from typing import Dict, List


class CSVExporter:
    """
    Exports test cases as CSV test case matrix
    """

    def __init__(self):
        """Initialize the CSV exporter"""
        self.headers = [
            'Test ID',
            'Test Name',
            'Description',
            'Category',
            'Steps',
            'Expected Result',
            'Assertions',
            'Priority'
        ]

    def export(
        self,
        tests: str,
        output_path: str,
        metadata: Dict = None
    ) -> str:
        """
        Export tests to CSV format

        Args:
            tests: Test code string
            output_path: Path to save the CSV file
            metadata: Additional metadata

        Returns:
            Path to the exported file
        """
        # Parse tests into rows
        test_rows = self._parse_tests_to_rows(tests)

        # Ensure .csv extension
        output_path = Path(output_path)
        if output_path.suffix != '.csv':
            output_path = output_path.with_suffix('.csv')

        # Write to CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open('w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.headers)
            writer.writeheader()
            writer.writerows(test_rows)

        return str(output_path)

    def _parse_tests_to_rows(self, tests: str) -> List[Dict]:
        """
        Parse test code into CSV rows

        Args:
            tests: Test code string

        Returns:
            List of row dictionaries
        """
        rows = []

        # Split into individual test functions
        test_pattern = r'def (test_\w+)\([^)]*\):(.*?)(?=\ndef |\Z)'
        matches = re.findall(test_pattern, tests, re.DOTALL)

        for idx, (test_name, test_body) in enumerate(matches, 1):
            # Extract docstring
            docstring_match = re.search(r'"""(.*?)"""', test_body, re.DOTALL)
            description = docstring_match.group(1).strip() if docstring_match else "No description"

            # Extract assertions
            assertions = re.findall(r'assert .+', test_body)
            assertion_text = '; '.join(assertions) if assertions else 'No assertions'

            # Determine category
            category = self._determine_category(test_name, test_body)

            # Extract steps (simplified)
            steps = self._extract_steps(test_body)

            # Extract expected result from assertions
            expected_result = self._extract_expected_result(assertions)

            # Determine priority
            priority = self._determine_priority(category, test_body)

            row = {
                'Test ID': f'TC-{idx:03d}',
                'Test Name': test_name,
                'Description': description,
                'Category': category,
                'Steps': steps,
                'Expected Result': expected_result,
                'Assertions': assertion_text,
                'Priority': priority
            }

            rows.append(row)

        return rows

    def _determine_category(self, test_name: str, test_body: str) -> str:
        """
        Determine test category based on name and content

        Args:
            test_name: Test function name
            test_body: Test function body

        Returns:
            Category string
        """
        test_name_lower = test_name.lower()
        test_body_lower = test_body.lower()

        if 'security' in test_name_lower or 'injection' in test_name_lower or 'xss' in test_name_lower:
            return 'Security'
        elif 'error' in test_name_lower or 'exception' in test_name_lower or 'pytest.raises' in test_body:
            return 'Error Handling'
        elif 'edge' in test_name_lower or 'boundary' in test_name_lower or 'empty' in test_name_lower:
            return 'Edge Case'
        elif 'valid' in test_name_lower or 'success' in test_name_lower or 'happy' in test_name_lower:
            return 'Happy Path'
        elif 'invalid' in test_name_lower or 'validation' in test_name_lower:
            return 'Validation'
        else:
            return 'Functional'

    def _extract_steps(self, test_body: str) -> str:
        """
        Extract test steps from test body

        Args:
            test_body: Test function body

        Returns:
            Steps as string
        """
        # Look for Arrange/Act/Assert pattern
        steps = []

        if '# Arrange' in test_body or '# arrange' in test_body.lower():
            steps.append('1. Setup test data')

        if '# Act' in test_body or '# act' in test_body.lower():
            steps.append('2. Execute function under test')

        if '# Assert' in test_body or '# assert' in test_body.lower():
            steps.append('3. Verify results')

        return '; '.join(steps) if steps else 'Execute test'

    def _extract_expected_result(self, assertions: List[str]) -> str:
        """
        Extract expected result from assertions

        Args:
            assertions: List of assert statements

        Returns:
            Expected result string
        """
        if not assertions:
            return 'Test passes'

        # Take the first assertion and extract the message if present
        first_assertion = assertions[0]

        # Look for assertion message
        message_match = re.search(r',\s*["\'](.+?)["\']', first_assertion)
        if message_match:
            return message_match.group(1)

        # Extract what's being asserted
        if '==' in first_assertion:
            return 'Values should be equal'
        elif 'is not None' in first_assertion:
            return 'Should not be None'
        elif 'is None' in first_assertion:
            return 'Should be None'
        elif '>' in first_assertion or '<' in first_assertion:
            return 'Should meet comparison criteria'
        else:
            return 'Assertion should pass'

    def _determine_priority(self, category: str, test_body: str) -> str:
        """
        Determine test priority

        Args:
            category: Test category
            test_body: Test function body

        Returns:
            Priority level (High/Medium/Low)
        """
        if category in ['Security', 'Happy Path']:
            return 'High'
        elif category in ['Error Handling', 'Validation']:
            return 'Medium'
        else:
            return 'Low'
