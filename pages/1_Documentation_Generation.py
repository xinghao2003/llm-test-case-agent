import streamlit as st
import requests
import json
import io
from datetime import datetime
import os

import markdown
import pdfkit

st.set_page_config(page_title="Test Case Documentation Generation",
                   page_icon="📈",
                   layout="wide")

# --- Configuration ---
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
# Common models that generally work well for text generation tasks.
# You can find more model IDs on OpenRouter: https://openrouter.ai/docs#models
AVAILABLE_MODELS = {
    "GPT-3.5 Turbo (OpenAI)": "openai/gpt-3.5-turbo",
    "Claude 3 Haiku (Anthropic)": "anthropic/claude-3-haiku-20240307",
    "Mistral 7B Instruct (MistralAI)": "mistralai/mistral-7b-instruct",
    "Llama 3 8B Instruct (Meta)": "meta-llama/llama-3-8b-instruct",
    # Often free on OpenRouter
    "Google: Gemma 3n 4B (free)": "google/gemma-3n-e4b-it:free",
    "Meta: Llama 3.3 8B Instruct (free)":
    "meta-llama/llama-3.3-8b-instruct:free",
    "Mistral: DevStral Small (free)": "mistralai/devstral-small:free"
}
DEFAULT_MODEL = "GPT-3.5 Turbo (OpenAI)"

# --- Helper Function to Call OpenRouter ---


def generate_test_cases_from_story(api_key, model_id, user_story,
                                   custom_prompt_template):
    """
    Sends a user story to OpenRouter and returns generated test cases.
    """
    if not api_key:
        st.error(
            "OpenRouter API Key is not set. Please add it to your secrets or enter it below."
        )
        return None
    if not user_story:
        st.warning("Please enter a user story.")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Construct the prompt using the template
    prompt_content = custom_prompt_template.format(user_story=user_story)

    data = {
        "model":
        model_id,
        "messages": [{
            "role":
            "system",
            "content":
            "You are an expert QA assistant. Your task is to generate detailed test cases from a given user story."
        }, {
            "role": "user",
            "content": prompt_content
        }]
    }

    try:
        response = requests.post(
            f"{OPENROUTER_API_BASE}/chat/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=120  # Increased timeout for longer generations
        )
        response.raise_for_status()  # Raise an exception for HTTP errors
        completion = response.json()
        if completion.get("choices") and len(completion["choices"]) > 0:
            message = completion["choices"][0].get("message",
                                                   {}).get("content")
            if message:
                return message.strip()
            else:
                st.error(
                    f"Error: LLM returned an empty message. Response: {completion}"
                )
                return None
        else:
            st.error(
                f"Error: No choices found in LLM response. Response: {completion}"
            )
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
        st.error(
            f"Response content: {response.text if 'response' in locals() else 'No response object'}"
        )
        return None
    except json.JSONDecodeError as e:
        st.error(f"JSON Decode Error: {e}")
        st.error(
            f"Response content that failed to parse: {response.text if 'response' in locals() else 'No response object'}"
        )
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None


# --- Helper Function to Generate PDF from Markdown ---
def generate_pdf_from_markdown(test_cases_content, user_story):
    """
    Generate a PDF from markdown content using pdfkit.
    """
    # Combine user story and test cases as markdown
    markdown_content = f"# Test Cases Report\n\n" \
                       f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n" \
                       f"## User Story\n\n" \
                       f"{user_story}\n\n" \
                       f"## Generated Test Cases\n\n" \
                       f"{test_cases_content}"

    # Convert markdown to HTML
    html_content = markdown.markdown(markdown_content,
                                     extensions=['tables', 'fenced_code'])

    # Convert HTML to PDF using pdfkit
    pdf_buffer = io.BytesIO()
    # pdfkit.from_string requires a file-like object for output
    pdfkit.from_string(html_content, False, output_path=pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer


# --- Streamlit UI ---
st.title("📈 Test Case Documentation Generation")
st.markdown(
    "Generate comprehensive test cases from your user stories using OpenRouter LLMs."
)

# --- API Key Input ---
# Try to get API key from secrets, otherwise allow user input
try:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
except FileNotFoundError:  # For environments where secrets.toml might not exist initially
    api_key = ""

if not api_key:
    st.warning(
        "OpenRouter API Key not found in secrets.toml. Please enter it below.")
    api_key_input = st.text_input("Enter your OpenRouter API Key:",
                                  type="password",
                                  key="api_key_input_field")
    if api_key_input:
        api_key = api_key_input
else:
    st.sidebar.success("OpenRouter API Key loaded from secrets.")

# --- Model Selection ---
st.sidebar.header("⚙️ Configuration")
selected_model_name = st.sidebar.selectbox(
    "Select LLM Model:",
    options=list(AVAILABLE_MODELS.keys()),
    index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL))
selected_model_id = AVAILABLE_MODELS[selected_model_name]
st.sidebar.caption(f"Using OpenRouter model ID: `{selected_model_id}`")

# --- Prompt Customization ---
st.sidebar.subheader("Prompt Customization")
default_prompt_template = """
Please generate comprehensive test cases for the following user story.
For each test case, include:
- Test Case ID: (e.g., TC_FEATURE_001)
- Description: (Brief objective of the test)
- Preconditions: (Any setup required before test execution, if applicable)
- Steps to Reproduce: (Numbered steps)
- Expected Result: (What should happen if the test passes)
- Test Type: (e.g., Positive, Negative, Boundary, UI, API, Performance, Security, Usability)

User Story:
---
{user_story}
---

Generated Test Cases:
"""
custom_prompt = st.sidebar.text_area(
    "Edit Prompt Template:",
    value=default_prompt_template,
    height=300,
    help="Use {user_story} as a placeholder for the input user story.")

# --- Main Area: User Story Input and Output ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Enter User Story")
    user_story_input = st.text_area(
        "Paste your user story here:",
        height=200,
        placeholder=
        "As a [type of user], I want [an action] so that [a benefit/value].")
    example_story = "As a registered user, I want to be able to reset my password using my email address, so that I can regain access to my account if I forget my password."
    if st.button("Load Example User Story"):
        user_story_input = example_story
        # Workaround to update text_area with a button click
        st.experimental_rerun()

    if st.button("🚀 Generate Test Cases",
                 type="primary",
                 use_container_width=True,
                 disabled=(not api_key or not user_story_input)):
        if not api_key:
            st.error("API Key is missing. Cannot generate.")
        elif not user_story_input:
            st.warning("Please enter a user story above.")
        else:
            with st.spinner(
                    f"Generating test cases using {selected_model_name}... This may take a moment."
            ):
                generated_cases = generate_test_cases_from_story(
                    api_key, selected_model_id, user_story_input,
                    custom_prompt)
                st.session_state.generated_cases = generated_cases  # Store in session state

with col2:
    st.subheader("Generated Test Cases")
    if "generated_cases" in st.session_state and st.session_state.generated_cases:
        st.markdown(st.session_state.generated_cases)

        # Action buttons
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            if st.button("📋 Copy to Clipboard"):
                # Shows it in a copyable block
                st.code(st.session_state.generated_cases)
                st.success("Test cases displayed in a copyable block above!")

        with col2_2:
            if st.button("📄 Export as PDF"):
                try:
                    pdf_buffer = generate_pdf_from_markdown(
                        st.session_state.generated_cases, user_story_input)
                    pdf_filename = f"test_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

                    st.download_button(label="💾 Download PDF",
                                       data=pdf_buffer,
                                       file_name=pdf_filename,
                                       mime="application/pdf",
                                       use_container_width=True)
                    st.success(
                        "PDF generated successfully! Click the download button above."
                    )
                except ImportError:
                    st.error(
                        "PDF generation requires the 'markdown' and 'pdfkit' libraries. Please install them using: pip install markdown pdfkit"
                    )
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")

    elif "generated_cases" in st.session_state and st.session_state.generated_cases is None and user_story_input:
        # This handles the case where generation was attempted but failed
        st.info(
            "Generation failed or returned no content. Check error messages above or in the terminal."
        )
    else:
        st.info("Test cases will appear here after generation.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with [Streamlit](https://streamlit.io) & [OpenRouter](https://openrouter.ai)"
)
