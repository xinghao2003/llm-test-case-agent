# System Architecture

## Overview

The AI-Powered Test Case Generator follows a modular, agent-based architecture with clear separation of concerns.

## Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                    │
│  ┌────────────────┐              ┌──────────────────────┐   │
│  │  Chat Interface│              │  Visualization       │   │
│  │  - Messages    │              │  - Coverage Charts   │   │
│  │  - User Input  │              │  - Progress Graphs   │   │
│  └────────────────┘              └──────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Test Generation Orchestrator                    │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Agentic Loop Controller                           │     │
│  │  - Manages iteration lifecycle                     │     │
│  │  - Coordinates agent components                    │     │
│  │  - Handles stop conditions                         │     │
│  │  - Manages user interaction callbacks              │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│     Test     │    │   Coverage   │    │     Test     │
│  Generator   │    │   Analyzer   │    │  Validator   │
│              │    │              │    │              │
│ - Initial    │    │ - Scenario   │    │ - Syntax     │
│   Generation │    │   Extraction │    │   Check      │
│ - Iterative  │    │ - Gap        │    │ - Pytest     │
│   Improvement│    │   Detection  │    │   Compat.    │
│ - Refinement │    │ - Self-      │    │ - BLEU       │
│ - Security   │    │   Critique   │    │   Score      │
│   Focus      │    │ - Coverage   │    │ - Quality    │
│              │    │   Scoring    │    │   Metrics    │
└──────────────┘    └──────────────┘    └──────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │   OpenRouter     │
                    │   LLM API        │
                    │   (GPT-4)        │
                    └──────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Prompt Templates                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Generation  │  │   Critique   │  │Clarification │      │
│  │   Prompts    │  │   Prompts    │  │   Prompts    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Exporters                               │
│  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐               │
│  │Python │  │ JSON  │  │  CSV  │  │  PDF  │               │
│  │  .py  │  │       │  │       │  │       │               │
│  └───────┘  └───────┘  └───────┘  └───────┘               │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Input Phase

```
User Story + Context
        │
        ▼
┌───────────────────┐
│  Orchestrator     │
│  - Validate input │
│  - Initialize     │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Ambiguity         │
│ Detection         │
│ (if interactive)  │
└───────────────────┘
        │
        ▼
   Clarifications
```

### 2. Generation Phase (Iteration Loop)

```
┌─────────────────────────────────────────────────────┐
│  FOR each iteration (1 to MAX_ITERATIONS):          │
│                                                      │
│  ┌────────────────────────────────────────────┐    │
│  │  1. Generate Tests                         │    │
│  │     - Initial or Additional                │    │
│  │     - Use appropriate prompts              │    │
│  │     - Extract Python code                  │    │
│  └────────────────────────────────────────────┘    │
│                     │                               │
│                     ▼                               │
│  ┌────────────────────────────────────────────┐    │
│  │  2. Validate Tests                         │    │
│  │     - Syntax checking                      │    │
│  │     - Pytest compatibility                 │    │
│  │     - Quality metrics                      │    │
│  └────────────────────────────────────────────┘    │
│                     │                               │
│                     ▼                               │
│  ┌────────────────────────────────────────────┐    │
│  │  3. Analyze Coverage                       │    │
│  │     - Extract scenarios                    │    │
│  │     - Identify covered/missing             │    │
│  │     - Calculate coverage score             │    │
│  └────────────────────────────────────────────┘    │
│                     │                               │
│                     ▼                               │
│  ┌────────────────────────────────────────────┐    │
│  │  4. Self-Critique                          │    │
│  │     - Evaluate quality                     │    │
│  │     - Identify weaknesses                  │    │
│  │     - Determine if continue                │    │
│  └────────────────────────────────────────────┘    │
│                     │                               │
│                     ▼                               │
│  ┌────────────────────────────────────────────┐    │
│  │  5. Check Stop Conditions                  │    │
│  │     - Coverage threshold?                  │    │
│  │     - Max iterations?                      │    │
│  │     - Improvement too small?               │    │
│  └────────────────────────────────────────────┘    │
│                     │                               │
│         ┌───────────┴──────────┐                   │
│         ▼                      ▼                   │
│      STOP                   CONTINUE               │
│                                │                   │
│  ┌────────────────────────────┘                   │
│  │  6. User Interaction (optional)                │
│  │     - Present progress                         │
│  │     - Ask clarifications                       │
│  │     - Get focus areas                          │
│  └────────────────────────────────────────────┐   │
│                     │                          │   │
│                     ▼                          │   │
│              Next Iteration ───────────────────┘   │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 3. Output Phase

```
Final Test Suite
        │
        ├──────────┬──────────┬──────────┐
        ▼          ▼          ▼          ▼
    Python      JSON       CSV        PDF
    (.py)
```

## Agent Components

### Orchestrator Agent

**Responsibilities:**
- Manages the overall agentic loop
- Coordinates sub-agents
- Tracks iteration state
- Implements stop conditions
- Handles callbacks for UI

**Key Methods:**
- `run(user_story, context, auto_mode)`: Main execution loop
- `should_continue_iteration()`: Decision logic
- `_merge_tests()`: Combine test suites

### Test Generator Agent

**Responsibilities:**
- LLM-based test generation
- Prompt construction
- Code extraction from LLM responses
- Multiple generation strategies

**Generation Modes:**
1. Initial: Fresh generation from user story
2. Iterative: Add tests to fill gaps
3. Refinement: Improve based on feedback
4. Security: Focus on security scenarios

**Key Methods:**
- `generate_from_user_story()`: Initial generation
- `generate_additional_tests()`: Gap-filling
- `refine_tests()`: User feedback integration
- `generate_security_tests()`: Security focus

### Coverage Analyzer Agent

**Responsibilities:**
- Scenario extraction
- Coverage measurement
- Gap detection
- Self-critique

**Analysis Types:**
1. Scenario Extraction: Parse user story for testable scenarios
2. Coverage Analysis: Compare tests to scenarios
3. Ambiguity Detection: Find unclear requirements
4. Self-Critique: Evaluate own performance

**Key Methods:**
- `analyze_coverage()`: Comprehensive coverage analysis
- `self_critique()`: Self-assessment
- `detect_ambiguities()`: Find unclear requirements
- `extract_scenarios()`: Parse user story

### Test Validator Agent

**Responsibilities:**
- Syntax validation
- Pytest compatibility checking
- BLEU score calculation
- Quality assessment

**Validation Checks:**
- Python syntax (AST parsing)
- Pytest conventions (naming, structure)
- Assertion presence
- Docstring presence
- Quality metrics

**Key Methods:**
- `validate_syntax()`: Python AST check
- `check_pytest_compatibility()`: Convention check
- `calculate_bleu_score()`: NLTK BLEU
- `evaluate()`: Comprehensive assessment

## Prompt Engineering Strategy

### Prompt Structure

All prompts follow this pattern:

```
1. Role Definition
   "You are an expert software testing engineer..."

2. Task Description
   "Your task is to generate pytest test functions..."

3. Input Context
   USER STORY: {user_story}
   CONTEXT: {context}
   EXISTING TESTS: {existing_tests}

4. Requirements
   - Generate valid Python pytest functions
   - Include docstrings
   - Use appropriate fixtures
   - Consider edge cases

5. Output Format
   - Specify exact format expected
   - Provide examples
   - Request JSON or code blocks

6. Examples (optional)
   Show example input/output
```

### Prompt Types

1. **Generation Prompts**
   - Initial generation: Broad, comprehensive
   - Iterative: Focused on gaps
   - Refinement: Incorporate feedback
   - Security: Specialized scenarios

2. **Critique Prompts**
   - Coverage analysis: Objective measurement
   - Self-critique: Critical evaluation
   - Ambiguity detection: Question formulation
   - Scenario extraction: Requirement parsing

3. **Clarification Prompts**
   - User-friendly language
   - Multiple choice when appropriate
   - Clear explanations
   - Allow open-ended responses

## State Management

### Session State (Streamlit)

```python
st.session_state = {
    'orchestrator': TestGenerationOrchestrator,
    'generation_complete': bool,
    'result': dict,
    'api_key': str,
    'current_iteration_data': list,
    'messages': list,  # Chat history
    'max_iterations': int,
    'coverage_threshold': float,
    'auto_mode': bool
}
```

### Result Structure

```python
result = {
    'tests': str,  # Final test code
    'coverage': float,  # Coverage score
    'iterations': int,  # Number of iterations
    'conversation_log': list,  # Chat history
    'iteration_history': list,  # Per-iteration data
    'metadata': {
        'final_validation': dict,
        'coverage_analysis': dict,
        'scenarios': dict,
        'elapsed_time': float,
        'timestamp': str
    }
}
```

## Extension Points

### Adding New Languages/Frameworks

1. **Create Generator Subclass**
```python
class JestTestGenerator(TestGenerator):
    def generate_from_user_story(self, story, context):
        # Use Jest-specific prompts
        prompt = JEST_GENERATION_PROMPT.format(...)
        # Generate describe/it blocks
```

2. **Add Framework-Specific Prompts**
```python
# prompts/jest_prompts.py
JEST_GENERATION_PROMPT = """
Generate Jest test suites with describe/it syntax...
"""
```

3. **Update Validators**
```python
class JestValidator(TestValidator):
    def validate_syntax(self, code):
        # Use JavaScript parser
        pass
```

### Adding New Metrics

1. **Extend Validator**
```python
class TestValidator:
    def calculate_rouge_score(self, generated, reference):
        # Implement ROUGE
        pass
```

2. **Update UI**
```python
# Display new metric
st.metric("ROUGE Score", f"{rouge:.2f}")
```

### Adding New Export Formats

1. **Create Exporter Class**
```python
class MarkdownExporter:
    def export(self, tests, output_path, metadata):
        # Generate markdown
        pass
```

2. **Register in UI**
```python
if st.button("Export Markdown"):
    exporter = MarkdownExporter()
    exporter.export(...)
```

## Performance Considerations

### Optimization Strategies

1. **Caching**
   - Cache LLM responses for identical inputs
   - Use `@st.cache_data` for expensive computations

2. **Parallel Processing**
   - Generate multiple test variants in parallel
   - Batch analyze multiple scenarios

3. **Streaming**
   - Stream LLM responses for real-time feedback
   - Progressive test display

4. **Rate Limiting**
   - Respect API rate limits
   - Implement exponential backoff

### Scalability

For production use:
- Add Redis for caching
- Use async/await for LLM calls
- Implement job queue for batch processing
- Add database for test history
- Horizontal scaling with load balancer

## Security Considerations

1. **API Key Management**
   - Store in environment variables
   - Never commit to version control
   - Use secrets management in production

2. **Input Validation**
   - Sanitize user story input
   - Validate context data
   - Prevent prompt injection

3. **Code Execution**
   - Never execute generated tests automatically
   - Warn users to review before running
   - Sandbox execution if needed

4. **Data Privacy**
   - Don't log sensitive user stories
   - Clear session data on logout
   - Comply with data protection regulations

## Testing Strategy

### Unit Tests
- Test each agent independently
- Mock LLM responses
- Validate prompt construction

### Integration Tests
- Test agent interactions
- Validate full iteration loop
- Check state management

### End-to-End Tests
- Test complete user workflows
- Validate UI interactions
- Check export functionality

## Future Architecture Improvements

1. **Microservices**
   - Separate LLM service
   - Dedicated validation service
   - Export service

2. **Event-Driven**
   - Use message queue
   - Async event processing
   - Better scalability

3. **Plugin System**
   - Dynamic agent loading
   - Custom prompt plugins
   - Third-party extensions

4. **Multi-Model Support**
   - Support multiple LLMs
   - A/B testing different models
   - Ensemble approaches
