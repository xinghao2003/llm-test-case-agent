import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.title("📝 User Story to Test Case Generator")
st.markdown("Welcome to the User Story to Test Case Generator application!")

st.markdown("""
## Features

- **Test Case Documentation Generation**: Generate comprehensive test cases from user stories using OpenRouter LLMs
- Export test cases as PDF documents
- Customize prompts for different testing scenarios

## Getting Started

1. Navigate to the **Test Case Documentation Generation** page from the sidebar
2. Enter your OpenRouter API key (if not configured in secrets)
3. Input your user story
4. Generate and export test cases

Built with [Streamlit](https://streamlit.io) & [OpenRouter](https://openrouter.ai)
""")
