"""
Self-critique and coverage analysis prompt templates
"""

COVERAGE_ANALYSIS_PROMPT = """You are a critical test reviewer analyzing test coverage for completeness and quality.

USER STORY:
{user_story}

GENERATED TESTS:
{tests}

Perform a comprehensive coverage analysis:

1. SCENARIO COVERAGE:
   - Identify all testable scenarios from the user story
   - List which scenarios are covered by the generated tests
   - List which scenarios are missing

2. QUALITY ASSESSMENT:
   - Are happy path scenarios tested?
   - Are edge cases covered (empty inputs, null values, boundary conditions)?
   - Are error conditions tested (exceptions, invalid inputs)?
   - Are security concerns addressed?
   - Is input validation comprehensive?

3. GAP IDENTIFICATION:
   - What specific test scenarios are missing?
   - What boundary values should be tested?
   - What error conditions aren't covered?
   - What security vulnerabilities aren't tested?

4. COVERAGE ESTIMATION:
   - Estimate coverage percentage (0.0 to 1.0)
   - Consider: scenario coverage, edge cases, error handling, security

OUTPUT FORMAT (JSON):
{{
  "coverage_score": <float between 0.0 and 1.0>,
  "total_scenarios": <int>,
  "covered_scenarios": [
    "scenario 1 description",
    "scenario 2 description"
  ],
  "missing_scenarios": [
    "missing scenario 1",
    "missing scenario 2"
  ],
  "gaps": [
    {{
      "category": "edge_case|error_handling|security|validation",
      "description": "specific gap description",
      "priority": "high|medium|low"
    }}
  ],
  "quality_issues": [
    "issue description if any"
  ],
  "recommendations": [
    "specific recommendation for improvement"
  ]
}}

Return ONLY valid JSON, no additional text.
"""

SELF_CRITIQUE_PROMPT = """You are critiquing your own test generation work. Be harsh and thorough in identifying weaknesses.

USER STORY:
{user_story}

YOUR GENERATED TESTS:
{tests}

ITERATION NUMBER: {iteration}

Critically evaluate your work:

1. What did you do well?
2. What did you miss or overlook?
3. Are there any weak or incomplete tests?
4. What assumptions did you make that might be wrong?
5. What edge cases did you not consider?
6. Are there any security concerns you didn't address?
7. Could any tests be more comprehensive?

Be specific and actionable in your critique.

OUTPUT FORMAT (JSON):
{{
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "missing_coverage": ["what's missing 1", "what's missing 2"],
  "improvement_suggestions": [
    {{
      "test_area": "specific area to improve",
      "suggestion": "concrete suggestion",
      "priority": "high|medium|low"
    }}
  ],
  "should_continue": <boolean>,
  "reasoning": "why continue or stop"
}}

Return ONLY valid JSON.
"""

AMBIGUITY_DETECTION_PROMPT = """You are analyzing a user story to detect ambiguities that need clarification before generating comprehensive tests.

USER STORY:
{user_story}

CONTEXT PROVIDED:
{context}

Analyze the user story for:
1. Unclear requirements
2. Missing specifications (data types, ranges, formats)
3. Ambiguous behavior descriptions
4. Undefined error handling
5. Missing acceptance criteria

For each ambiguity found, formulate a clear question for the user that:
- Explains what's unclear
- Provides 2-3 possible interpretations (if applicable)
- Allows for open-ended responses
- Is written in simple, non-technical language

OUTPUT FORMAT (JSON):
{{
  "ambiguities_found": <boolean>,
  "clarification_needed": [
    {{
      "topic": "brief topic name",
      "question": "clear question for user",
      "options": ["option A", "option B", "option C (or other)"],
      "importance": "critical|important|nice-to-have"
    }}
  ],
  "assumptions": [
    "assumption 1 we'll make if no clarification",
    "assumption 2 we'll make if no clarification"
  ]
}}

Return ONLY valid JSON.
"""

SCENARIO_EXTRACTION_PROMPT = """Extract all testable scenarios from the following user story.

USER STORY:
{user_story}

CONTEXT:
{context}

Identify:
1. Main success scenario (happy path)
2. Alternative flows
3. Error scenarios
4. Edge cases implied by the requirements
5. Security considerations
6. Validation requirements

OUTPUT FORMAT (JSON):
{{
  "happy_path": ["main success scenario"],
  "edge_cases": ["edge case 1", "edge case 2"],
  "error_scenarios": ["error 1", "error 2"],
  "validation_scenarios": ["validation 1", "validation 2"],
  "security_scenarios": ["security 1", "security 2"],
  "boundary_conditions": ["boundary 1", "boundary 2"]
}}

Return ONLY valid JSON.
"""
