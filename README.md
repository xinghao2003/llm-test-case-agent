# 🧪 AI-Powered User Story to Test Case Generator

A Proof-of-Concept (PoC) system that automatically generates Python pytest test cases from natural language user stories using agentic AI with self-iteration and human-in-the-loop conversations.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.9+-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## 🌟 Key Features

### 🤖 Agentic Behavior
- **Self-Iterating**: Automatically improves test coverage through multiple iterations
- **Self-Critique**: Analyzes its own test generation and identifies gaps
- **Coverage Analysis**: Tracks and measures test scenario coverage
- **Smart Stop Conditions**: Knows when to stop iterating (coverage threshold, max iterations, diminishing returns)

### 💬 Conversational UI
- **Human-in-the-Loop**: Pauses for clarification when needed
- **Ambiguity Detection**: Identifies unclear requirements and asks questions
- **Validation Checkpoints**: Presents results for user review
- **Interactive Chat**: Natural conversation flow with the AI agent

### 📊 Comprehensive Test Generation
- **Happy Path**: Main success scenarios
- **Edge Cases**: Boundary values, empty inputs, extreme cases
- **Error Handling**: Exception scenarios, invalid inputs
- **Security Tests**: SQL injection, XSS, authentication
- **Validation**: Input format checking, constraint validation

### 📤 Multiple Export Formats
- **Python (.py)**: Ready-to-run pytest files
- **JSON**: Structured test data for CI/CD
- **CSV**: Test case matrix for documentation
- **PDF**: Formatted test documentation

### 📈 Quality Metrics
- **BLEU Score**: Semantic similarity measurement
- **Syntax Validation**: Python AST parsing
- **Pytest Compatibility**: Convention checking
- **Coverage Visualization**: Interactive charts and graphs

## 🏗️ Architecture

```
joyce-fyp/
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration & settings
├── agents/                         # Agentic components
│   ├── orchestrator.py            # Main iteration loop controller
│   ├── test_generator.py          # LLM-based test generation
│   ├── coverage_analyzer.py       # Gap detection & self-critique
│   └── validator.py               # Quality assessment
├── prompts/                        # LLM prompt templates
│   ├── generation_prompts.py      # Test generation prompts
│   ├── critique_prompts.py        # Self-assessment prompts
│   └── clarification_prompts.py   # User interaction prompts
├── exporters/                      # Export functionality
│   ├── python_exporter.py         # .py export
│   ├── json_exporter.py           # JSON export
│   ├── csv_exporter.py            # CSV export
│   └── pdf_exporter.py            # PDF export
├── ui/                             # UI components
│   ├── chat_interface.py          # Conversational interface
│   └── visualization.py           # Coverage visualizations
├── parsers/                        # Utility parsers
├── examples/                       # Example user stories
└── exports/                        # Generated exports (gitignored)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- OpenRouter API key (for GPT-4 access)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/joyce-fyp.git
cd joyce-fyp
```

2. **Create a virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data (for BLEU score)**
```bash
python -c "import nltk; nltk.download('punkt')"
```

5. **Configure API key**

Create a `.env` file in the project root:
```bash
cp .env.example .env
```

Edit `.env` and add your OpenRouter API key:
```
OPENROUTER_API_KEY=your_api_key_here
```

Alternatively, you can enter the API key directly in the Streamlit sidebar when running the app.

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### Basic Workflow

1. **Enter User Story**
   - Type or paste your user story in the text area
   - Add any additional context or requirements
   - Or select an example from the sidebar

2. **Configure Settings** (Optional)
   - Adjust max iterations (default: 5)
   - Set coverage threshold (default: 80%)
   - Choose auto mode or interactive mode

3. **Generate Tests**
   - Click "Generate Tests"
   - Watch the AI agent work through iterations
   - Review generation log and progress

4. **Review Results**
   - View generated test code
   - Check coverage analysis
   - Review iteration history
   - Examine quality metrics

5. **Export**
   - Choose export format (Python, CSV, JSON, PDF)
   - Download or save to exports folder

### Example User Story

```
As a user, I want to register an account with email and password
so that I can access the platform.

Additional Context:
- Email must be valid format
- Password minimum 8 characters with numbers and special characters
- Prevent duplicate email registrations
```

### Interactive Mode

In interactive mode, the agent will pause to ask questions:

```
Agent: "I found ambiguity regarding password requirements.
       What's the minimum password length?"

You: "8 characters minimum"

Agent: "Should password history be checked?"

You: "No, not needed for this PoC"

Agent: "Generating refined tests with your clarifications..."
```

### Auto Mode

In auto mode, the agent runs autonomously:
- Makes reasonable assumptions
- Focuses on high-priority gaps
- Continues until coverage threshold or max iterations

## 🎯 Agentic Loop Details

### Iteration Process

```
1. Initial Generation
   ├─ Generate base tests from user story
   ├─ Validate syntax and pytest compatibility
   └─ Measure initial coverage

2. For each iteration (up to max):
   ├─ Analyze Coverage
   │  ├─ Identify covered scenarios
   │  ├─ Detect missing scenarios
   │  └─ Find gaps (edge cases, security, etc.)
   │
   ├─ Self-Critique
   │  ├─ Evaluate current test quality
   │  ├─ Identify weaknesses
   │  └─ Determine if iteration should continue
   │
   ├─ User Interaction (if not auto mode)
   │  ├─ Present current state
   │  ├─ Ask for clarifications
   │  └─ Get focus areas
   │
   ├─ Generate Additional Tests
   │  ├─ Address identified gaps
   │  ├─ Focus on priority areas
   │  └─ Merge with existing tests
   │
   └─ Check Stop Conditions
      ├─ Coverage threshold met?
      ├─ Max iterations reached?
      ├─ Improvement too small?
      └─ User requested stop?

3. Finalization
   ├─ Final validation
   ├─ Generate visualizations
   └─ Prepare for export
```

### Stop Conditions

The agent stops iterating when:
1. **Coverage Threshold**: Target coverage (e.g., 80%) is achieved
2. **Max Iterations**: Maximum number of iterations (e.g., 5) reached
3. **Diminishing Returns**: Coverage improvement < 5% from previous iteration
4. **User Request**: User manually stops the process

## 📊 Quality Metrics Explained

### Coverage Score
- Percentage of identified scenarios that have test coverage
- Calculated by: `covered_scenarios / total_scenarios`
- Goal: Reach 80%+ coverage

### BLEU Score
- Measures semantic similarity to reference tests (if available)
- Range: 0.0 to 1.0 (higher is better)
- Uses 4-gram BLEU with smoothing
- Note: Only available when reference tests are provided

### Validation Checks
- ✅ **Syntax Valid**: Python code parses without errors
- ✅ **Pytest Compatible**: Follows pytest conventions
- ✅ **Has Assertions**: Tests include assert statements
- ✅ **Has Docstrings**: Tests are documented
- ✅ **Has Fixtures**: Uses pytest fixtures (bonus)
- ✅ **Has Parametrize**: Uses parametrization (bonus)

## 🔧 Configuration

### Environment Variables (`.env`)

```bash
# Required
OPENROUTER_API_KEY=your_key_here

# Optional - Override defaults
MODEL_NAME=openai/gpt-4-turbo
MAX_ITERATIONS=5
COVERAGE_THRESHOLD=0.80
TEMPERATURE=0.7
MAX_TOKENS=2000
```

### Config File (`config.py`)

Key settings you can modify:
- `MAX_ITERATIONS`: Number of self-iteration cycles
- `COVERAGE_THRESHOLD`: Target coverage to achieve
- `MIN_COVERAGE_IMPROVEMENT`: Minimum improvement to continue
- `TEMPERATURE`: LLM sampling temperature
- `BLEU_WEIGHTS`: N-gram weights for BLEU calculation

## 🎨 Customization

### Adding Custom Prompts

Edit prompt templates in `prompts/`:
- `generation_prompts.py`: Customize test generation behavior
- `critique_prompts.py`: Adjust coverage analysis
- `clarification_prompts.py`: Modify user interaction style

### Custom Export Formats

Create a new exporter in `exporters/`:

```python
class CustomExporter:
    def export(self, tests: str, output_path: str, metadata: dict):
        # Your export logic here
        pass
```

Register in `exporters/__init__.py` and add to `app.py`.

### Extending to Other Languages/Frameworks

The architecture is designed for extension:

1. Create language-specific generator in `agents/`
2. Add framework-specific prompts in `prompts/`
3. Update `config.py` with language/framework options
4. Modify UI to support selection

Example for Jest (JavaScript):
```python
class JestTestGenerator(TestGenerator):
    def generate_from_user_story(self, story, context):
        # Use Jest-specific prompts
        # Generate describe/it blocks
        pass
```

## 📚 Examples

See `examples/` directory for:
- `example_user_registration.md`: Detailed user story example
- `example_generated_tests.py`: Sample generated test suite

## 🐛 Troubleshooting

### Common Issues

**Issue**: "API key not found"
- **Solution**: Set `OPENROUTER_API_KEY` in `.env` or enter in sidebar

**Issue**: "Module not found"
- **Solution**: Ensure all dependencies installed: `pip install -r requirements.txt`

**Issue**: "NLTK data not found"
- **Solution**: Download NLTK data: `python -c "import nltk; nltk.download('punkt')"`

**Issue**: "PDF export fails"
- **Solution**: Install wkhtmltopdf or use alternative PDF library

**Issue**: "Tests have syntax errors"
- **Solution**: Try regenerating with more context, or manually fix and iterate

### Debug Mode

Enable detailed logging in `config.py`:
```python
LOG_LEVEL = "DEBUG"
```

## 🔮 Future Enhancements

### Planned Features
- [ ] Support for JavaScript/TypeScript (Jest, Mocha)
- [ ] Support for Java (JUnit)
- [ ] Integration with JIRA for user story import
- [ ] GitHub Actions integration
- [ ] Test execution and actual coverage measurement
- [ ] Mutation testing support
- [ ] Test prioritization by risk
- [ ] Flaky test detection
- [ ] Multi-language UI
- [ ] Team collaboration features

### Scalability Considerations
- Add caching for repeated test generation
- Batch processing for multiple user stories
- Database storage for test history
- API endpoint for CI/CD integration

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [OpenRouter](https://openrouter.ai/) & GPT-4
- Uses [pytest](https://pytest.org/) conventions
- Visualizations with [Plotly](https://plotly.com/)

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review example files in `examples/`

## 🎓 Academic Use

This project was created as a Final Year Project (FYP) demonstrating:
- Agentic AI systems
- Human-in-the-loop machine learning
- Automated test generation
- Self-iterating algorithms
- Conversational AI interfaces

**Citation**:
```
@software{ai_test_generator_2024,
  author = {Your Name},
  title = {AI-Powered User Story to Test Case Generator},
  year = {2024},
  url = {https://github.com/yourusername/joyce-fyp}
}
```

---

**Built with ❤️ for better software testing**
