"""
Coverage analysis and self-critique system
"""
import json
import re
from typing import Dict, List, Optional
from openai import OpenAI
import config
from prompts.critique_prompts import (
    COVERAGE_ANALYSIS_PROMPT,
    SELF_CRITIQUE_PROMPT,
    AMBIGUITY_DETECTION_PROMPT,
    SCENARIO_EXTRACTION_PROMPT
)


class CoverageAnalyzer:
    """
    Analyzes test coverage and identifies gaps using LLM-based critique
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the coverage analyzer

        Args:
            api_key: OpenRouter API key (defaults to config)
        """
        self.api_key = api_key or config.OPENROUTER_API_KEY
        self.client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=self.api_key
        )
        self.model = config.MODEL_NAME

    def _call_llm(self, prompt: str, temperature: float = 0.3) -> str:
        """
        Make a call to the LLM API for analysis (lower temperature for consistency)

        Args:
            prompt: The analysis prompt
            temperature: Sampling temperature

        Returns:
            LLM response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a critical test coverage analyzer. Provide thorough, objective analysis in JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Coverage analysis API call failed: {str(e)}")

    def _extract_json(self, response: str) -> Dict:
        """
        Extract JSON from LLM response

        Args:
            response: Raw LLM response

        Returns:
            Parsed JSON dict
        """
        # Try to find JSON in code blocks
        json_block_pattern = r"```json\n(.*?)\n```"
        matches = re.findall(json_block_pattern, response, re.DOTALL)

        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass

        # Try to extract JSON directly
        try:
            # Find first { and last }
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # If all else fails, return a default structure
        return {
            "coverage_score": 0.5,
            "covered_scenarios": [],
            "missing_scenarios": ["Unable to parse coverage analysis"],
            "gaps": [],
            "recommendations": []
        }

    def analyze_coverage(
        self,
        user_story: str,
        tests: str
    ) -> Dict:
        """
        Analyze test coverage comprehensively

        Args:
            user_story: Original user story
            tests: Generated test code

        Returns:
            Dict with coverage analysis:
            {
                'coverage_score': float (0.0-1.0),
                'total_scenarios': int,
                'covered_scenarios': List[str],
                'missing_scenarios': List[str],
                'gaps': List[Dict],
                'quality_issues': List[str],
                'recommendations': List[str]
            }
        """
        prompt = COVERAGE_ANALYSIS_PROMPT.format(
            user_story=user_story,
            tests=tests
        )

        response = self._call_llm(prompt)
        analysis = self._extract_json(response)

        # Ensure required fields exist
        analysis.setdefault('coverage_score', 0.5)
        analysis.setdefault('total_scenarios', len(analysis.get('covered_scenarios', [])) + len(analysis.get('missing_scenarios', [])))
        analysis.setdefault('covered_scenarios', [])
        analysis.setdefault('missing_scenarios', [])
        analysis.setdefault('gaps', [])
        analysis.setdefault('quality_issues', [])
        analysis.setdefault('recommendations', [])

        return analysis

    def self_critique(
        self,
        user_story: str,
        tests: str,
        iteration: int
    ) -> Dict:
        """
        Perform self-critique of generated tests

        Args:
            user_story: Original user story
            tests: Generated test code
            iteration: Current iteration number

        Returns:
            Dict with critique:
            {
                'strengths': List[str],
                'weaknesses': List[str],
                'missing_coverage': List[str],
                'improvement_suggestions': List[Dict],
                'should_continue': bool,
                'reasoning': str
            }
        """
        prompt = SELF_CRITIQUE_PROMPT.format(
            user_story=user_story,
            tests=tests,
            iteration=iteration
        )

        response = self._call_llm(prompt, temperature=0.4)
        critique = self._extract_json(response)

        # Ensure required fields
        critique.setdefault('strengths', [])
        critique.setdefault('weaknesses', [])
        critique.setdefault('missing_coverage', [])
        critique.setdefault('improvement_suggestions', [])
        critique.setdefault('should_continue', iteration < config.MAX_ITERATIONS)
        critique.setdefault('reasoning', 'Continue iterating to improve coverage')

        return critique

    def detect_ambiguities(
        self,
        user_story: str,
        context: str = ""
    ) -> Dict:
        """
        Detect ambiguities in user story that need clarification

        Args:
            user_story: The user story text
            context: Additional context provided

        Returns:
            Dict with ambiguities:
            {
                'ambiguities_found': bool,
                'clarification_needed': List[Dict],
                'assumptions': List[str]
            }
        """
        prompt = AMBIGUITY_DETECTION_PROMPT.format(
            user_story=user_story,
            context=context or "No additional context provided."
        )

        response = self._call_llm(prompt, temperature=0.3)
        ambiguities = self._extract_json(response)

        # Ensure required fields
        ambiguities.setdefault('ambiguities_found', False)
        ambiguities.setdefault('clarification_needed', [])
        ambiguities.setdefault('assumptions', [])

        return ambiguities

    def extract_scenarios(
        self,
        user_story: str,
        context: str = ""
    ) -> Dict:
        """
        Extract all testable scenarios from user story

        Args:
            user_story: The user story text
            context: Additional context

        Returns:
            Dict with scenarios by category:
            {
                'happy_path': List[str],
                'edge_cases': List[str],
                'error_scenarios': List[str],
                'validation_scenarios': List[str],
                'security_scenarios': List[str],
                'boundary_conditions': List[str]
            }
        """
        prompt = SCENARIO_EXTRACTION_PROMPT.format(
            user_story=user_story,
            context=context or "No additional context provided."
        )

        response = self._call_llm(prompt, temperature=0.3)
        scenarios = self._extract_json(response)

        # Ensure all categories exist
        scenarios.setdefault('happy_path', [])
        scenarios.setdefault('edge_cases', [])
        scenarios.setdefault('error_scenarios', [])
        scenarios.setdefault('validation_scenarios', [])
        scenarios.setdefault('security_scenarios', [])
        scenarios.setdefault('boundary_conditions', [])

        return scenarios

    def calculate_coverage_score(
        self,
        covered_scenarios: List[str],
        total_scenarios: int
    ) -> float:
        """
        Calculate a simple coverage score

        Args:
            covered_scenarios: List of covered scenario descriptions
            total_scenarios: Total number of identified scenarios

        Returns:
            Coverage score between 0.0 and 1.0
        """
        if total_scenarios == 0:
            return 0.0

        return min(len(covered_scenarios) / total_scenarios, 1.0)

    def identify_priority_gaps(
        self,
        gaps: List[Dict]
    ) -> List[Dict]:
        """
        Sort gaps by priority

        Args:
            gaps: List of gap dicts with 'priority' field

        Returns:
            Sorted list (high priority first)
        """
        priority_order = {'high': 0, 'medium': 1, 'low': 2}

        return sorted(
            gaps,
            key=lambda g: priority_order.get(g.get('priority', 'low'), 2)
        )

    def should_continue_iteration(
        self,
        coverage_score: float,
        iteration: int,
        previous_score: float = 0.0
    ) -> Dict[str, any]:
        """
        Determine if iteration should continue

        Args:
            coverage_score: Current coverage score
            iteration: Current iteration number
            previous_score: Previous iteration's coverage score

        Returns:
            Dict with 'should_continue' and 'reason'
        """
        # Stop if max iterations reached
        if iteration >= config.MAX_ITERATIONS:
            return {
                'should_continue': False,
                'reason': f'Maximum iterations ({config.MAX_ITERATIONS}) reached'
            }

        # Stop if coverage threshold met
        if coverage_score >= config.COVERAGE_THRESHOLD:
            return {
                'should_continue': False,
                'reason': f'Coverage threshold ({config.COVERAGE_THRESHOLD:.0%}) achieved'
            }

        # Stop if improvement is too small
        improvement = coverage_score - previous_score
        if iteration > 1 and improvement < config.MIN_COVERAGE_IMPROVEMENT:
            return {
                'should_continue': False,
                'reason': f'Coverage improvement too small ({improvement:.1%})'
            }

        # Continue iterating
        return {
            'should_continue': True,
            'reason': f'Coverage at {coverage_score:.0%}, continuing to improve'
        }
