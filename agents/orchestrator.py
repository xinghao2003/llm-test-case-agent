"""
Main orchestrator for agentic test generation with self-iteration
"""
import time
from typing import Dict, List, Optional, Callable
from datetime import datetime
import config
from .test_generator import TestGenerator
from .coverage_analyzer import CoverageAnalyzer
from .validator import TestValidator


class TestGenerationOrchestrator:
    """
    Manages the agentic iteration loop for test case generation
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the orchestrator with all required agents

        Args:
            api_key: OpenRouter API key
        """
        self.generator = TestGenerator(api_key)
        self.analyzer = CoverageAnalyzer(api_key)
        self.validator = TestValidator()

        self.conversation_history = []
        self.iteration_history = []
        self.current_tests = ""
        self.coverage_scores = []

        # Configuration
        self.max_iterations = config.MAX_ITERATIONS
        self.coverage_threshold = config.COVERAGE_THRESHOLD

        # Callbacks for UI interaction
        self.on_message = None  # Callback for agent messages
        self.on_request_input = None  # Callback for user input requests
        self.on_iteration_complete = None  # Callback for iteration updates

    def set_callbacks(
        self,
        on_message: Callable = None,
        on_request_input: Callable = None,
        on_iteration_complete: Callable = None
    ):
        """
        Set callback functions for UI interaction

        Args:
            on_message: Called when agent has a message (agent_message: str)
            on_request_input: Called when agent needs user input (prompt: str) -> str
            on_iteration_complete: Called after each iteration (iteration_data: dict)
        """
        self.on_message = on_message
        self.on_request_input = on_request_input
        self.on_iteration_complete = on_iteration_complete

    def _send_message(self, message: str, message_type: str = "info"):
        """Send a message to the UI"""
        if self.on_message:
            self.on_message(message, message_type)
        else:
            print(f"[{message_type.upper()}] {message}")

    def _request_user_input(self, prompt: str, input_type: str = "text") -> str:
        """Request input from the user"""
        if self.on_request_input:
            return self.on_request_input(prompt, input_type)
        else:
            return input(f"{prompt}\n> ")

    def _notify_iteration_complete(self, iteration_data: Dict):
        """Notify that an iteration is complete"""
        if self.on_iteration_complete:
            self.on_iteration_complete(iteration_data)

    def run(
        self,
        user_story: str,
        additional_context: str = "",
        auto_mode: bool = False
    ) -> Dict:
        """
        Main agentic loop with self-iteration

        Args:
            user_story: The user story to generate tests for
            additional_context: Additional context or requirements
            auto_mode: If True, run without user interaction

        Returns:
            Dict containing:
            {
                'tests': str,  # Final test code
                'coverage': float,  # Final coverage score
                'iterations': int,  # Number of iterations performed
                'conversation_log': List[Dict],  # Full conversation history
                'iteration_history': List[Dict],  # History of each iteration
                'metadata': Dict  # Additional metadata
            }
        """
        start_time = time.time()
        self._send_message(f"🚀 Starting test generation for user story...", "info")

        # Step 1: Detect ambiguities (optional - can be skipped in auto mode)
        if not auto_mode:
            self._send_message("🔍 Analyzing user story for ambiguities...", "info")
            ambiguities = self.analyzer.detect_ambiguities(user_story, additional_context)

            if ambiguities.get('ambiguities_found', False):
                clarifications_needed = ambiguities.get('clarification_needed', [])

                if clarifications_needed:
                    self._send_message(
                        f"⚠️ Found {len(clarifications_needed)} ambiguities that need clarification",
                        "warning"
                    )

                    # Ask user for clarifications
                    for clarification in clarifications_needed:
                        if clarification.get('importance') == 'critical':
                            question = f"**{clarification['topic']}**\n\n{clarification['question']}\n"
                            if 'options' in clarification and clarification['options']:
                                question += "\nOptions:\n" + "\n".join(
                                    f"  {i+1}. {opt}"
                                    for i, opt in enumerate(clarification['options'])
                                )

                            user_response = self._request_user_input(question)
                            additional_context += f"\n\n{clarification['topic']}: {user_response}"

        # Step 2: Extract scenarios from user story
        self._send_message("📋 Extracting testable scenarios...", "info")
        scenarios = self.analyzer.extract_scenarios(user_story, additional_context)

        total_scenarios = sum([
            len(scenarios.get('happy_path', [])),
            len(scenarios.get('edge_cases', [])),
            len(scenarios.get('error_scenarios', [])),
            len(scenarios.get('validation_scenarios', [])),
            len(scenarios.get('security_scenarios', []))
        ])

        self._send_message(f"✅ Identified {total_scenarios} testable scenarios", "success")

        # Step 3: Initial test generation
        self._send_message("🔨 Generating initial test cases...", "info")

        initial_result = self.generator.generate_from_user_story(
            user_story,
            additional_context
        )

        self.current_tests = initial_result['code']
        iteration = 1

        # Validate initial tests
        validation = self.validator.evaluate(self.current_tests)

        if not validation['syntax_valid']:
            self._send_message(
                f"⚠️ Syntax error in generated tests: {validation['syntax_error']}",
                "error"
            )

        self._send_message(
            f"✅ Generated {validation['test_count']} initial tests",
            "success"
        )

        # Iteration loop
        previous_coverage = 0.0

        while iteration <= self.max_iterations:
            self._send_message(f"\n🔄 Iteration {iteration}/{self.max_iterations}", "info")

            # Analyze current coverage
            self._send_message("📊 Analyzing coverage...", "info")
            coverage_analysis = self.analyzer.analyze_coverage(user_story, self.current_tests)

            current_coverage = coverage_analysis['coverage_score']
            self.coverage_scores.append(current_coverage)

            self._send_message(
                f"📈 Coverage: {current_coverage:.1%} ({len(coverage_analysis['covered_scenarios'])} scenarios covered)",
                "info"
            )

            # Store iteration data
            iteration_data = {
                'iteration': iteration,
                'tests': self.current_tests,
                'coverage': current_coverage,
                'covered_scenarios': coverage_analysis['covered_scenarios'],
                'missing_scenarios': coverage_analysis['missing_scenarios'],
                'gaps': coverage_analysis['gaps'],
                'validation': validation
            }
            self.iteration_history.append(iteration_data)
            self._notify_iteration_complete(iteration_data)

            # Check if we should continue
            should_continue = self.analyzer.should_continue_iteration(
                current_coverage,
                iteration,
                previous_coverage
            )

            if not should_continue['should_continue']:
                self._send_message(f"🎯 {should_continue['reason']}", "success")
                break

            # Self-critique
            critique = self.analyzer.self_critique(user_story, self.current_tests, iteration)

            # Ask user if they want to continue (if not auto mode)
            if not auto_mode and iteration > 1:
                continue_prompt = f"""
**Iteration {iteration} Summary:**
- Coverage: {current_coverage:.1%}
- Missing: {len(coverage_analysis['missing_scenarios'])} scenarios
- Gaps identified: {len(coverage_analysis['gaps'])}

Should I continue iterating? (yes/no/focus)
- yes: Continue with automatic improvements
- no: Stop and finalize tests
- focus: Let me specify what to focus on
"""
                user_choice = self._request_user_input(continue_prompt)

                if user_choice.lower().startswith('n'):
                    self._send_message("⏹️ Stopping iteration as requested", "info")
                    break
                elif user_choice.lower().startswith('f'):
                    focus_prompt = "What should I focus on? (e.g., 'security', 'edge cases', 'error handling')"
                    focus_input = self._request_user_input(focus_prompt)
                    focus_areas = [f.strip() for f in focus_input.split(',')]
                else:
                    focus_areas = None
            else:
                # Auto mode: focus on high-priority gaps
                focus_areas = [
                    gap['category']
                    for gap in coverage_analysis['gaps']
                    if gap.get('priority') == 'high'
                ]

            # Generate additional tests
            self._send_message("🔨 Generating additional tests to fill gaps...", "info")

            additional_result = self.generator.generate_additional_tests(
                user_story,
                self.current_tests,
                coverage_analysis['gaps'],
                focus_areas
            )

            # Merge new tests with existing
            new_tests = additional_result['code']

            # Simple merge: append new tests
            if new_tests.strip():
                # Extract only the new test functions (skip duplicate imports)
                self.current_tests = self._merge_tests(self.current_tests, new_tests)

                # Validate merged tests
                validation = self.validator.evaluate(self.current_tests)

                self._send_message(
                    f"✅ Added {additional_result['metadata']['test_count']} new tests",
                    "success"
                )
            else:
                self._send_message("⚠️ No new tests generated", "warning")

            previous_coverage = current_coverage
            iteration += 1

        # Final validation and summary
        final_validation = self.validator.evaluate(self.current_tests)
        final_coverage = coverage_analysis['coverage_score']

        elapsed_time = time.time() - start_time

        self._send_message("\n✨ Test generation complete!", "success")
        self._send_message(f"📊 Final coverage: {final_coverage:.1%}", "info")
        self._send_message(f"🧪 Total tests: {final_validation['test_count']}", "info")
        self._send_message(f"🔄 Iterations: {iteration - 1}", "info")
        self._send_message(f"⏱️ Time: {elapsed_time:.1f}s", "info")

        return {
            'tests': self.current_tests,
            'coverage': final_coverage,
            'iterations': iteration - 1,
            'conversation_log': self.conversation_history,
            'iteration_history': self.iteration_history,
            'metadata': {
                'final_validation': final_validation,
                'coverage_analysis': coverage_analysis,
                'scenarios': scenarios,
                'elapsed_time': elapsed_time,
                'timestamp': datetime.now().isoformat()
            }
        }

    def _merge_tests(self, existing: str, new: str) -> str:
        """
        Merge new tests with existing tests, avoiding duplicates

        Args:
            existing: Existing test code
            new: New test code to merge

        Returns:
            Merged test code
        """
        # Extract imports from both
        existing_imports = set()
        new_imports = set()

        for line in existing.split('\n'):
            if line.strip().startswith(('import ', 'from ')):
                existing_imports.add(line.strip())

        for line in new.split('\n'):
            if line.strip().startswith(('import ', 'from ')):
                new_imports.add(line.strip())

        # Combine imports
        all_imports = sorted(existing_imports | new_imports)

        # Remove imports from both and combine
        existing_without_imports = '\n'.join([
            line for line in existing.split('\n')
            if not line.strip().startswith(('import ', 'from '))
        ])

        new_without_imports = '\n'.join([
            line for line in new.split('\n')
            if not line.strip().startswith(('import ', 'from '))
        ])

        # Combine
        merged = '\n'.join(all_imports) + '\n\n' + existing_without_imports.strip() + '\n\n' + new_without_imports.strip()

        return merged.strip()
