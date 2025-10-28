"""
Test validation and quality assessment using BLEU score and syntax checking
"""
import ast
import re
from typing import Dict, List, Optional
import config


class TestValidator:
    """
    Validates generated test code for syntax, pytest compatibility, and quality
    """

    def __init__(self):
        """Initialize the validator"""
        self.bleu_weights = config.BLEU_WEIGHTS

    def validate_syntax(self, code: str) -> Dict[str, any]:
        """
        Check if generated code is syntactically valid Python

        Args:
            code: Python code string

        Returns:
            Dict with 'valid', 'error', and 'line' information
        """
        try:
            ast.parse(code)
            return {
                'valid': True,
                'error': None,
                'line': None
            }
        except SyntaxError as e:
            return {
                'valid': False,
                'error': str(e),
                'line': e.lineno
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'line': None
            }

    def check_pytest_compatibility(self, code: str) -> Dict[str, any]:
        """
        Check if code follows pytest conventions

        Args:
            code: Python code string

        Returns:
            Dict with compatibility checks
        """
        checks = {
            'has_test_functions': bool(re.search(r'\ndef test_', code)),
            'has_pytest_import': 'pytest' in code,
            'uses_assertions': 'assert' in code,
            'has_docstrings': '"""' in code or "'''" in code,
            'uses_fixtures': '@pytest.fixture' in code,
            'uses_markers': '@pytest.mark' in code,
            'has_parametrize': '@pytest.mark.parametrize' in code
        }

        # Count test functions
        test_count = len(re.findall(r'\ndef test_', code))
        checks['test_count'] = test_count

        # Overall compatibility
        checks['compatible'] = (
            checks['has_test_functions'] and
            checks['uses_assertions']
        )

        # Quality score (0-1)
        quality_factors = [
            checks['has_test_functions'],
            checks['has_pytest_import'],
            checks['uses_assertions'],
            checks['has_docstrings'],
        ]
        checks['quality_score'] = sum(quality_factors) / len(quality_factors)

        return checks

    def calculate_bleu_score(
        self,
        generated: str,
        reference: Optional[str] = None
    ) -> float:
        """
        Calculate BLEU score between generated and reference tests

        Note: For PoC, we use a simplified BLEU implementation
        For production, use nltk.translate.bleu_score

        Args:
            generated: Generated test code
            reference: Reference test code (if available)

        Returns:
            BLEU score (0.0-1.0) or -1 if no reference
        """
        if not reference:
            return -1.0  # No reference available

        try:
            # Try to import NLTK for proper BLEU calculation
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from nltk.tokenize import word_tokenize

            # Tokenize
            reference_tokens = [word_tokenize(reference.lower())]
            generated_tokens = word_tokenize(generated.lower())

            # Calculate BLEU with smoothing
            smoothing = SmoothingFunction()
            score = sentence_bleu(
                reference_tokens,
                generated_tokens,
                weights=self.bleu_weights,
                smoothing_function=smoothing.method1
            )

            return score

        except ImportError:
            # Fallback: simple word overlap metric
            ref_words = set(reference.lower().split())
            gen_words = set(generated.lower().split())

            if not ref_words:
                return 0.0

            overlap = len(ref_words & gen_words)
            return overlap / len(ref_words)

    def evaluate(
        self,
        generated_tests: str,
        reference_tests: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Comprehensive evaluation of generated tests

        Args:
            generated_tests: Generated test code
            reference_tests: Reference test code (optional)

        Returns:
            Dict with evaluation metrics:
            {
                'syntax_valid': bool,
                'pytest_compatible': bool,
                'bleu_score': float,
                'quality_score': float,
                'test_count': int,
                'has_assertions': bool,
                'has_docstrings': bool,
                'quality_issues': List[str],
                'recommendations': List[str]
            }
        """
        # Syntax validation
        syntax_result = self.validate_syntax(generated_tests)

        # Pytest compatibility
        pytest_checks = self.check_pytest_compatibility(generated_tests)

        # BLEU score (if reference available)
        bleu_score = self.calculate_bleu_score(generated_tests, reference_tests)

        # Identify quality issues
        quality_issues = []
        recommendations = []

        if not syntax_result['valid']:
            quality_issues.append(f"Syntax error: {syntax_result['error']}")
            recommendations.append("Fix syntax errors before using tests")

        if not pytest_checks['has_pytest_import']:
            quality_issues.append("Missing pytest import")
            recommendations.append("Add 'import pytest' at the top of the file")

        if not pytest_checks['has_docstrings']:
            quality_issues.append("Tests lack docstrings")
            recommendations.append("Add docstrings to explain test purpose")

        if pytest_checks['test_count'] == 0:
            quality_issues.append("No test functions found")
            recommendations.append("Ensure functions start with 'test_'")

        if pytest_checks['test_count'] < 3:
            quality_issues.append("Limited test coverage (< 3 tests)")
            recommendations.append("Consider adding more test scenarios")

        # Overall quality assessment
        overall_quality = pytest_checks['quality_score']
        if syntax_result['valid']:
            overall_quality = (overall_quality + 1.0) / 2  # Average with syntax check

        return {
            'syntax_valid': syntax_result['valid'],
            'syntax_error': syntax_result.get('error'),
            'pytest_compatible': pytest_checks['compatible'],
            'bleu_score': bleu_score,
            'quality_score': overall_quality,
            'test_count': pytest_checks['test_count'],
            'has_assertions': pytest_checks['uses_assertions'],
            'has_docstrings': pytest_checks['has_docstrings'],
            'has_fixtures': pytest_checks['uses_fixtures'],
            'has_parametrize': pytest_checks['has_parametrize'],
            'quality_issues': quality_issues,
            'recommendations': recommendations
        }

    def extract_test_names(self, code: str) -> List[str]:
        """
        Extract all test function names from code

        Args:
            code: Python test code

        Returns:
            List of test function names
        """
        pattern = r'def (test_\w+)\('
        return re.findall(pattern, code)

    def count_assertions(self, code: str) -> int:
        """
        Count number of assertions in test code

        Args:
            code: Python test code

        Returns:
            Number of assert statements
        """
        return len(re.findall(r'\bassert\b', code))

    def analyze_test_structure(self, code: str) -> Dict[str, any]:
        """
        Analyze the structure of test code

        Args:
            code: Python test code

        Returns:
            Dict with structure analysis
        """
        return {
            'test_names': self.extract_test_names(code),
            'test_count': len(self.extract_test_names(code)),
            'assertion_count': self.count_assertions(code),
            'avg_assertions_per_test': (
                self.count_assertions(code) / max(len(self.extract_test_names(code)), 1)
            ),
            'has_setup': 'def setup' in code.lower() or '@pytest.fixture' in code,
            'has_teardown': 'def teardown' in code.lower(),
            'uses_context_manager': 'with ' in code,
            'uses_exception_testing': 'pytest.raises' in code
        }
