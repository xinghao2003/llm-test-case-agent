"""
Conversational chat interface components
"""
import streamlit as st
from typing import Dict, List, Optional


class ChatInterface:
    """
    Manages conversational UI for human-in-the-loop interaction
    """

    def __init__(self):
        """Initialize the chat interface"""
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize Streamlit session state for chat"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'waiting_for_input' not in st.session_state:
            st.session_state.waiting_for_input = False
        if 'input_prompt' not in st.session_state:
            st.session_state.input_prompt = ""
        if 'user_response' not in st.session_state:
            st.session_state.user_response = None

    def add_message(
        self,
        content: str,
        role: str = "assistant",
        message_type: str = "info",
        metadata: Dict = None
    ):
        """
        Add a message to the chat history

        Args:
            content: Message content
            role: 'user' or 'assistant'
            message_type: 'info', 'success', 'warning', 'error'
            metadata: Additional metadata to store
        """
        message = {
            'role': role,
            'content': content,
            'type': message_type,
            'metadata': metadata or {}
        }
        st.session_state.messages.append(message)

    def display_messages(self):
        """Display all messages in the chat history"""
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                self._render_message_content(message)

    def _render_message_content(self, message: Dict):
        """
        Render a message with appropriate styling

        Args:
            message: Message dict
        """
        content = message['content']
        message_type = message.get('type', 'info')

        # Apply styling based on type
        if message_type == 'success':
            st.success(content)
        elif message_type == 'warning':
            st.warning(content)
        elif message_type == 'error':
            st.error(content)
        else:
            st.markdown(content)

        # Display metadata if present
        metadata = message.get('metadata', {})
        if metadata.get('show_code'):
            with st.expander("View Code", expanded=False):
                st.code(metadata['code'], language='python')

    def get_user_input(
        self,
        prompt: str,
        input_type: str = "text",
        options: List[str] = None,
        key: str = None
    ) -> Optional[str]:
        """
        Get input from the user

        Args:
            prompt: Prompt to display
            input_type: 'text', 'select', 'multiselect', 'confirm'
            options: Options for select/multiselect
            key: Unique key for the input widget

        Returns:
            User's input or None if not provided yet
        """
        # Generate unique key if not provided
        if key is None:
            key = f"input_{len(st.session_state.messages)}"

        if input_type == "text":
            return st.text_area(prompt, key=key, height=100)

        elif input_type == "select":
            if options:
                return st.selectbox(prompt, options, key=key)
            return None

        elif input_type == "multiselect":
            if options:
                return st.multiselect(prompt, options, key=key)
            return None

        elif input_type == "confirm":
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes", key=f"{key}_yes"):
                    return "yes"
            with col2:
                if st.button("No", key=f"{key}_no"):
                    return "no"
            return None

        return None

    def display_agent_thinking(self, message: str):
        """
        Display agent "thinking" indicator

        Args:
            message: Thinking message
        """
        with st.chat_message("assistant"):
            with st.spinner(message):
                st.empty()

    def display_code_preview(
        self,
        code: str,
        title: str = "Generated Tests",
        allow_edit: bool = False
    ) -> Optional[str]:
        """
        Display code with syntax highlighting and optional editing

        Args:
            code: Code to display
            title: Title for the code block
            allow_edit: Whether to allow editing

        Returns:
            Edited code if allow_edit=True, otherwise None
        """
        st.subheader(title)

        if allow_edit:
            edited_code = st.text_area(
                "Edit the generated tests:",
                value=code,
                height=400,
                key=f"code_edit_{len(st.session_state.messages)}"
            )
            return edited_code
        else:
            st.code(code, language='python', line_numbers=True)
            return None

    def display_progress(
        self,
        current: int,
        total: int,
        label: str = "Progress"
    ):
        """
        Display progress bar

        Args:
            current: Current progress value
            total: Total value
            label: Progress label
        """
        progress = current / total if total > 0 else 0
        st.progress(progress, text=f"{label}: {current}/{total}")

    def display_iteration_status(
        self,
        iteration: int,
        max_iterations: int,
        coverage: float,
        test_count: int
    ):
        """
        Display current iteration status

        Args:
            iteration: Current iteration number
            max_iterations: Maximum iterations
            coverage: Current coverage
            test_count: Current test count
        """
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Iteration", f"{iteration}/{max_iterations}")

        with col2:
            st.metric("Coverage", f"{coverage:.1%}")

        with col3:
            st.metric("Tests", test_count)

    def create_options_menu(
        self,
        title: str,
        options: List[Dict[str, str]],
        key: str = None
    ) -> Optional[str]:
        """
        Create an options menu for user choice

        Args:
            title: Menu title
            options: List of dicts with 'label' and 'value' keys
            key: Unique key for the widget

        Returns:
            Selected option value
        """
        st.subheader(title)

        for idx, option in enumerate(options):
            if st.button(
                option['label'],
                key=f"{key}_{idx}" if key else f"option_{idx}",
                use_container_width=True
            ):
                return option['value']

        return None

    def display_clarification_request(
        self,
        topic: str,
        question: str,
        options: List[str] = None,
        importance: str = "normal"
    ):
        """
        Display a clarification request to the user

        Args:
            topic: Topic of clarification
            question: Question to ask
            options: Optional list of suggested options
            importance: 'critical', 'important', or 'normal'
        """
        # Choose emoji based on importance
        emoji = "🚨" if importance == "critical" else "❓"

        with st.chat_message("assistant"):
            st.markdown(f"{emoji} **{topic}**")
            st.markdown(question)

            if options:
                st.markdown("**Suggested options:**")
                for idx, option in enumerate(options, 1):
                    st.markdown(f"{idx}. {option}")

    def clear_chat(self):
        """Clear the chat history"""
        st.session_state.messages = []
        st.session_state.waiting_for_input = False
        st.session_state.input_prompt = ""
        st.session_state.user_response = None

    def export_chat_history(self) -> str:
        """
        Export chat history as markdown

        Returns:
            Markdown-formatted chat history
        """
        lines = ["# Test Generation Chat History\n"]

        for message in st.session_state.messages:
            role = "**Agent**" if message['role'] == 'assistant' else "**User**"
            lines.append(f"\n{role}:")
            lines.append(message['content'])
            lines.append("")

        return "\n".join(lines)
