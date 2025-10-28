"""
LLM-based pytest test case generator
"""
import json
import re
from typing import Dict, List, Optional
from openai import OpenAI
import config
from prompts.generation_prompts import (
    INITIAL_GENERATION_PROMPT,
    ITERATIVE_GENERATION_PROMPT,
    REFINEMENT_PROMPT,
    SECURITY_FOCUSED_PROMPT,
    CODEBASE_AWARE_GENERATION_PROMPT,
    CODEBASE_ITERATIVE_PROMPT
)


class TestGenerator:
    """
    Generates pytest test functions using LLM
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the test generator with OpenRouter API

        Args:
            api_key: OpenRouter API key (defaults to config)
        """
        self.api_key = api_key or config.OPENROUTER_API_KEY
        self.client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=self.api_key
        )
        self.model = config.MODEL_NAME
        self.temperature = config.TEMPERATURE
        self.max_tokens = config.MAX_TOKENS

    def _call_llm(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        Make a call to the LLM API

        Args:
            prompt: The prompt to send
            temperature: Override default temperature

        Returns:
            LLM response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert software testing engineer specializing in Python and pytest. Generate high-quality, comprehensive test cases."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature or self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"LLM API call failed: {str(e)}")

    def _extract_code_from_response(self, response: str) -> str:
        """
        Extract Python code from LLM response (handles markdown code blocks)

        Args:
            response: Raw LLM response

        Returns:
            Extracted Python code
        """
        # Try to extract from markdown code blocks
        code_block_pattern = r"```python\n(.*?)\n```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            return "\n\n".join(matches)

        # If no code blocks, try to extract Python code directly
        # Remove common non-code prefixes
        lines = response.split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            # Skip explanation lines
            if line.strip().startswith(('Here', 'The following', 'I have', 'These tests')):
                continue
            # Start collecting when we see import or def
            if line.strip().startswith(('import ', 'from ', 'def ', '@pytest', 'class ')):
                in_code = True
            if in_code:
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines)

        # Return as-is if no patterns match
        return response

    def generate_from_user_story(
        self,
        user_story: str,
        additional_context: str = ""
    ) -> Dict[str, any]:
        """
        Generate initial pytest test functions from a user story

        Args:
            user_story: The user story describing functionality
            additional_context: Additional context or requirements

        Returns:
            Dict with 'code', 'raw_response', and 'metadata'
        """
        prompt = INITIAL_GENERATION_PROMPT.format(
            user_story=user_story,
            additional_context=additional_context or "No additional context provided."
        )

        raw_response = self._call_llm(prompt)
        code = self._extract_code_from_response(raw_response)

        # Extract test count
        test_count = len(re.findall(r'\ndef test_', code))

        return {
            'code': code,
            'raw_response': raw_response,
            'metadata': {
                'test_count': test_count,
                'generation_type': 'initial',
                'has_imports': 'import' in code,
                'has_pytest_markers': '@pytest' in code
            }
        }

    def generate_additional_tests(
        self,
        user_story: str,
        existing_tests: str,
        gaps: List[Dict],
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Generate additional tests to fill coverage gaps

        Args:
            user_story: Original user story
            existing_tests: Previously generated test code
            gaps: List of identified coverage gaps
            focus_areas: Specific areas to focus on (e.g., 'security', 'edge_cases')

        Returns:
            Dict with 'code', 'raw_response', and 'metadata'
        """
        # Format gaps for the prompt
        gaps_text = "\n".join([
            f"- {gap.get('category', 'general')}: {gap.get('description', '')}"
            for gap in gaps
        ])

        focus_text = ", ".join(focus_areas) if focus_areas else "general coverage improvement"

        prompt = ITERATIVE_GENERATION_PROMPT.format(
            user_story=user_story,
            existing_tests=existing_tests,
            gaps=gaps_text,
            focus_areas=focus_text
        )

        raw_response = self._call_llm(prompt)
        code = self._extract_code_from_response(raw_response)

        test_count = len(re.findall(r'\ndef test_', code))

        return {
            'code': code,
            'raw_response': raw_response,
            'metadata': {
                'test_count': test_count,
                'generation_type': 'iterative',
                'gaps_addressed': len(gaps),
                'focus_areas': focus_areas
            }
        }

    def refine_tests(
        self,
        original_tests: str,
        user_feedback: str,
        clarifications: Dict[str, str]
    ) -> Dict[str, any]:
        """
        Refine existing tests based on user feedback and clarifications

        Args:
            original_tests: Current test code
            user_feedback: User's feedback or requests
            clarifications: Dict of clarification questions and answers

        Returns:
            Dict with refined 'code', 'raw_response', and 'metadata'
        """
        clarifications_text = "\n".join([
            f"Q: {q}\nA: {a}"
            for q, a in clarifications.items()
        ])

        prompt = REFINEMENT_PROMPT.format(
            original_tests=original_tests,
            user_feedback=user_feedback,
            clarifications=clarifications_text
        )

        raw_response = self._call_llm(prompt)
        code = self._extract_code_from_response(raw_response)

        test_count = len(re.findall(r'\ndef test_', code))

        return {
            'code': code,
            'raw_response': raw_response,
            'metadata': {
                'test_count': test_count,
                'generation_type': 'refinement',
                'clarifications_applied': len(clarifications)
            }
        }

    def generate_security_tests(
        self,
        user_story: str,
        existing_tests: str
    ) -> Dict[str, any]:
        """
        Generate security-focused tests

        Args:
            user_story: Original user story
            existing_tests: Current test code

        Returns:
            Dict with security test 'code', 'raw_response', and 'metadata'
        """
        prompt = SECURITY_FOCUSED_PROMPT.format(
            user_story=user_story,
            existing_tests=existing_tests
        )

        raw_response = self._call_llm(prompt, temperature=0.5)  # Lower temp for security
        code = self._extract_code_from_response(raw_response)

        test_count = len(re.findall(r'\ndef test_', code))

        return {
            'code': code,
            'raw_response': raw_response,
            'metadata': {
                'test_count': test_count,
                'generation_type': 'security',
                'focus': 'security_scenarios'
            }
        }

    def merge_test_suites(self, test_suites: List[str]) -> str:
        """
        Merge multiple test code blocks, removing duplicates

        Args:
            test_suites: List of test code strings

        Returns:
            Merged test code
        """
        imports = set()
        tests = []
        seen_test_names = set()

        for suite in test_suites:
            lines = suite.split('\n')

            for line in lines:
                # Collect imports
                if line.strip().startswith(('import ', 'from ')):
                    imports.add(line.strip())
                # Collect test functions (avoid duplicates)
                elif line.strip().startswith('def test_'):
                    test_name = line.split('(')[0].replace('def ', '').strip()
                    if test_name not in seen_test_names:
                        seen_test_names.add(test_name)
                        # Find the complete function
                        test_start = suite.find(line)
                        if test_start != -1:
                            # Extract function (simplified - could be improved)
                            tests.append(line)

        # Combine
        merged = '\n'.join(sorted(imports)) + '\n\n' + '\n\n'.join(tests)
        return merged

    def generate_from_codebase(
        self,
        user_story: str,
        codebase_context: str,
        additional_context: str = ""
    ) -> Dict[str, any]:
        """
        Generate tests from user story with actual codebase analysis

        Args:
            user_story: The user story describing functionality
            codebase_context: Formatted codebase information (functions, classes, etc.)
            additional_context: Additional requirements

        Returns:
            Dict with 'code', 'raw_response', and 'metadata'
        """
        prompt = CODEBASE_AWARE_GENERATION_PROMPT.format(
            user_story=user_story,
            additional_context=additional_context or "No additional context provided.",
            codebase_context=codebase_context
        )

        raw_response = self._call_llm(prompt)
        code = self._extract_code_from_response(raw_response)

        test_count = len(re.findall(r'\ndef test_', code))

        return {
            'code': code,
            'raw_response': raw_response,
            'metadata': {
                'test_count': test_count,
                'generation_type': 'codebase_aware',
                'has_imports': 'import' in code,
                'has_real_imports': 'from ' in code
            }
        }

    def generate_additional_tests_from_codebase(
        self,
        user_story: str,
        existing_tests: str,
        codebase_context: str,
        gaps: List[Dict],
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Generate additional tests to fill gaps, using codebase context

        Args:
            user_story: Original user story
            existing_tests: Previously generated test code
            codebase_context: Formatted codebase information
            gaps: List of identified coverage gaps
            focus_areas: Specific areas to focus on

        Returns:
            Dict with 'code', 'raw_response', and 'metadata'
        """
        # Format gaps for the prompt
        gaps_text = "\n".join([
            f"- {gap.get('category', 'general')}: {gap.get('description', '')}"
            for gap in gaps
        ])

        focus_text = ", ".join(focus_areas) if focus_areas else "general coverage improvement"

        prompt = CODEBASE_ITERATIVE_PROMPT.format(
            user_story=user_story,
            existing_tests=existing_tests,
            codebase_context=codebase_context,
            gaps=gaps_text,
            focus_areas=focus_text
        )

        raw_response = self._call_llm(prompt)
        code = self._extract_code_from_response(raw_response)

        test_count = len(re.findall(r'\ndef test_', code))

        return {
            'code': code,
            'raw_response': raw_response,
            'metadata': {
                'test_count': test_count,
                'generation_type': 'codebase_iterative',
                'gaps_addressed': len(gaps),
                'focus_areas': focus_areas
            }
        }
