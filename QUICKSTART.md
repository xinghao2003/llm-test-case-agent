# Quick Start Guide

Get up and running with the AI Test Case Generator in 5 minutes!

## ⚡ Super Quick Start

```bash
# 1. Clone and navigate
git clone <your-repo-url>
cd joyce-fyp

# 2. Run setup script
chmod +x setup.sh
./setup.sh

# 3. Add API key to .env
echo "OPENROUTER_API_KEY=your_key_here" > .env

# 4. Run the app
source venv/bin/activate
streamlit run app.py
```

## 📝 First Test Generation

1. **Open the app** (it should open automatically at http://localhost:8501)

2. **Enter a simple user story** (or use an example from sidebar):
   ```
   As a user, I want to login with username and password
   ```

3. **Add context** (optional):
   ```
   Username: 3-20 characters, alphanumeric
   Password: minimum 8 characters
   ```

4. **Click "Generate Tests"** and watch the magic happen!

5. **Review the results** in the tabs:
   - Generated Tests: Your pytest code
   - Coverage Analysis: What's covered
   - Iteration History: How it improved
   - Quality Metrics: Validation results

6. **Export** your tests in your preferred format

## 🎯 Example User Stories

### Simple Example
```
As a customer, I want to add items to my cart
so that I can purchase multiple products at once.

Context:
- Items have name, price, and quantity
- Cart should calculate total price
- Quantity must be positive integer
```

### More Complex Example
```
As an admin, I want to export user data to CSV
so that I can analyze user behavior.

Context:
- Include: username, email, signup_date, last_login
- Filter by date range
- File size limit: 10MB
- Email should be anonymized if user opts out
```

## ⚙️ Settings to Try

### Quick Generation (Fast)
- Max Iterations: 2
- Coverage Threshold: 60%
- Auto Mode: ON

### Comprehensive Coverage (Thorough)
- Max Iterations: 5
- Coverage Threshold: 85%
- Auto Mode: ON

### Interactive Mode (Learning)
- Max Iterations: 3
- Coverage Threshold: 75%
- Auto Mode: OFF
  (The agent will ask you questions!)

## 🔍 What to Look For

### In Generated Tests
- ✅ Clear test names (test_feature_scenario)
- ✅ Docstrings explaining purpose
- ✅ Arrange-Act-Assert structure
- ✅ Good assertions with messages

### In Coverage Analysis
- 📊 Coverage gauge should be green
- ✅ Most scenarios listed as "Covered"
- 📈 Coverage improving each iteration

### In Quality Metrics
- ✓ Syntax Valid
- ✓ Pytest Compatible
- ✓ Has Assertions
- ✓ Has Docstrings

## 🎨 Tips for Better Results

1. **Be Specific**: More details = better tests
   - ❌ "User can search"
   - ✅ "User can search products by name, case-insensitive, with autocomplete"

2. **Include Constraints**: Help the AI understand boundaries
   - "Password: 8-128 characters"
   - "Age: 18-120 years"
   - "File size: max 5MB"

3. **Mention Error Cases**: What should fail?
   - "Reject duplicate emails"
   - "Return 404 if user not found"
   - "Validate credit card format"

4. **Add Security Requirements**: Get security tests
   - "Prevent SQL injection"
   - "Sanitize HTML inputs"
   - "Rate limit: 10 requests/minute"

## 🐛 Quick Troubleshooting

**Problem**: "API key not found"
```bash
# Solution: Check .env file
cat .env
# Should show: OPENROUTER_API_KEY=sk-...
```

**Problem**: "Module not found"
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt
```

**Problem**: "Tests have syntax errors"
```
Solution:
1. Try adding more context to your user story
2. Re-generate with focus on specific areas
3. Manually fix small issues in the generated code
```

**Problem**: "Coverage stuck at low %"
```
Solution:
1. Increase max iterations
2. Lower coverage threshold
3. Add more detail to user story
4. Use interactive mode to provide clarifications
```

## 📚 Next Steps

1. **Try More Examples**: Use the example stories in the sidebar
2. **Experiment with Settings**: Adjust iterations and coverage
3. **Export Tests**: Try different export formats
4. **Read README.md**: Learn about advanced features
5. **Check ARCHITECTURE.md**: Understand how it works

## 💡 Pro Tips

### Iteration Strategy
- Start with 2-3 iterations to see how it works
- Increase to 5+ for complex features
- Use interactive mode for ambiguous requirements

### Focus Areas
When agent asks what to focus on:
- "security" → SQL injection, XSS, auth tests
- "edge cases" → Empty inputs, boundaries, nulls
- "error handling" → Exceptions, invalid inputs
- "validation" → Format checking, constraints

### Best Practices
1. Review generated tests before using
2. Customize prompts for your domain
3. Keep user stories focused (one feature per story)
4. Export to Python and add to your test suite
5. Run with pytest to verify they work

## 🎓 Learning Path

**Beginner**:
1. Use example user stories
2. Auto mode ON
3. Just observe the generation

**Intermediate**:
1. Write your own user stories
2. Try interactive mode
3. Provide clarifications

**Advanced**:
1. Customize prompts (prompts/*.py)
2. Adjust configuration (config.py)
3. Extend for other frameworks

## 🚀 Your First Session

Here's what a typical first session looks like:

```
⏱️ 0:00 - Open app, enter API key
⏱️ 0:30 - Select "User Registration" example
⏱️ 1:00 - Click "Generate Tests"
⏱️ 1:30 - Watch iteration 1 complete
⏱️ 2:00 - Watch iteration 2 complete
⏱️ 2:30 - Review generated tests (12 tests, 85% coverage!)
⏱️ 3:00 - Check coverage analysis
⏱️ 3:30 - Export to Python file
⏱️ 4:00 - Open exported file in editor
⏱️ 4:30 - Run with pytest (they pass!)
⏱️ 5:00 - Feeling like a testing wizard! 🧙‍♂️
```

## 📞 Get Help

- **Questions?** Check the [README](README.md)
- **Issues?** See [Troubleshooting](#-quick-troubleshooting)
- **Want to learn more?** Read [ARCHITECTURE](ARCHITECTURE.md)
- **Have feedback?** Open an issue on GitHub

---

**Ready? Let's generate some tests!** 🧪✨

```bash
streamlit run app.py
```
