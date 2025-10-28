"""
User clarification and interaction prompt templates
"""

CLARIFICATION_REQUEST_TEMPLATE = """I need clarification on the following aspect of your user story:

**Topic**: {topic}

**Question**: {question}

{options_section}

**Why this matters**: {importance_explanation}

Please provide your input or select from the options above.
"""

VALIDATION_CHECKPOINT_TEMPLATE = """I've generated **{test_count}** test cases after **{iteration_count}** iteration(s).

**Coverage Status**: {coverage_percentage}%

**Scenarios Covered**:
{covered_scenarios}

**Current Tests**:
{test_summary}

**Please review and let me know**:
1. Do these tests align with your expectations?
2. Are there specific scenarios you'd like me to focus on?
3. Should I continue iterating for more comprehensive coverage?

You can:
- ✅ Approve and export
- 🔄 Request more iterations
- ✏️ Suggest specific scenarios to add
- 🎯 Focus on particular areas (security, edge cases, etc.)
"""

ITERATION_SUMMARY_TEMPLATE = """**Iteration {iteration_number} Complete**

**What I added this iteration**:
{new_tests_summary}

**Coverage improvement**: {previous_coverage}% → {current_coverage}% (+{improvement}%)

**Remaining gaps**:
{remaining_gaps}

**Next steps**: {next_action}
"""

COMPLETION_SUMMARY_TEMPLATE = """**Test Generation Complete! 🎉**

**Final Statistics**:
- Total tests generated: {total_tests}
- Iterations performed: {iterations}
- Final coverage: {coverage}%
- Time elapsed: {elapsed_time}

**Coverage Breakdown**:
- ✅ Happy path: {happy_path_coverage}
- ✅ Edge cases: {edge_case_coverage}
- ✅ Error handling: {error_coverage}
- ✅ Security: {security_coverage}

**Export Options**:
1. 📄 Python file (.py) - Ready to run with pytest
2. 📊 CSV - Test case matrix
3. 📋 JSON - Structured test data
4. 📕 PDF - Formatted documentation

What would you like to do next?
"""

FOCUS_AREA_PROMPT = """I can focus on specific types of tests in the next iteration. Please choose your priority:

1. **Security Testing** - SQL injection, XSS, authentication, authorization
2. **Edge Cases** - Boundary values, empty inputs, null values, extreme values
3. **Error Handling** - Exception cases, invalid inputs, system errors
4. **Performance** - Load testing, response time, resource usage
5. **Integration** - API interactions, database operations, external services
6. **User Input Validation** - Format checking, constraint validation, sanitization

Enter the number(s) or describe what you'd like me to focus on:
"""

AMBIGUITY_CLARIFICATION_TEMPLATE = """I detected some ambiguities in the user story that could affect test quality:

{ambiguities_list}

**I'll proceed with these assumptions if you don't provide clarification**:
{assumptions_list}

Would you like to clarify any of these points, or should I proceed with the assumptions?
"""
