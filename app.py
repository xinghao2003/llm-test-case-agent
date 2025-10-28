"""
AI-Powered User Story to Test Case Generator PoC
Main Streamlit Application
"""
import streamlit as st
import os
from pathlib import Path
from datetime import datetime
import traceback

# Import components
import config
from agents import TestGenerationOrchestrator
from exporters import PythonExporter, JSONExporter, CSVExporter, PDFExporter
from ui import ChatInterface, CoverageVisualizer

# Page configuration
st.set_page_config(
    page_title=config.STREAMLIT_PAGE_TITLE,
    page_icon=config.STREAMLIT_PAGE_ICON,
    layout=config.STREAMLIT_LAYOUT,
    initial_sidebar_state="expanded"
)


class TestGeneratorApp:
    """Main application class"""

    def __init__(self):
        """Initialize the application"""
        self.chat = ChatInterface()
        self.viz = CoverageVisualizer()
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'orchestrator' not in st.session_state:
            st.session_state.orchestrator = None

        if 'generation_complete' not in st.session_state:
            st.session_state.generation_complete = False

        if 'result' not in st.session_state:
            st.session_state.result = None

        if 'api_key' not in st.session_state:
            st.session_state.api_key = config.OPENROUTER_API_KEY

        if 'current_iteration_data' not in st.session_state:
            st.session_state.current_iteration_data = []

    def render_sidebar(self):
        """Render the sidebar with configuration and examples"""
        with st.sidebar:
            st.title("⚙️ Configuration")

            # API Key input
            api_key = st.text_input(
                "OpenRouter API Key",
                value=st.session_state.api_key,
                type="password",
                help="Enter your OpenRouter API key"
            )

            if api_key:
                st.session_state.api_key = api_key
                os.environ['OPENROUTER_API_KEY'] = api_key

            st.divider()

            # Generation settings
            st.subheader("Generation Settings")

            max_iterations = st.slider(
                "Max Iterations",
                min_value=1,
                max_value=10,
                value=config.MAX_ITERATIONS,
                help="Maximum number of self-iteration cycles"
            )

            coverage_threshold = st.slider(
                "Coverage Threshold",
                min_value=0.5,
                max_value=1.0,
                value=config.COVERAGE_THRESHOLD,
                step=0.05,
                format="%.0f%%",
                help="Target coverage percentage to stop iteration"
            )

            auto_mode = st.checkbox(
                "Auto Mode",
                value=True,
                help="Run without user interaction during iterations"
            )

            st.session_state.max_iterations = max_iterations
            st.session_state.coverage_threshold = coverage_threshold
            st.session_state.auto_mode = auto_mode

            st.divider()

            # Example user stories
            st.subheader("📝 Example User Stories")

            for example in config.EXAMPLE_STORIES:
                if st.button(example['title'], use_container_width=True):
                    st.session_state.example_story = example['story']
                    st.session_state.example_context = example['context']
                    st.rerun()

            st.divider()

            # Clear button
            if st.button("🗑️ Clear All", use_container_width=True):
                self.chat.clear_chat()
                st.session_state.generation_complete = False
                st.session_state.result = None
                st.session_state.current_iteration_data = []
                if 'example_story' in st.session_state:
                    del st.session_state.example_story
                if 'example_context' in st.session_state:
                    del st.session_state.example_context
                st.rerun()

            # Info
            st.divider()
            st.caption("AI-Powered Test Case Generator v1.0")
            st.caption("Built with Streamlit & GPT-4")

    def render_main_interface(self):
        """Render the main interface"""
        st.title("🧪 AI-Powered Test Case Generator")
        st.markdown("Generate comprehensive pytest test cases from user stories using agentic AI")

        # Check for API key
        if not st.session_state.api_key:
            st.warning("⚠️ Please enter your OpenRouter API key in the sidebar to get started.")
            st.info("Don't have an API key? Get one at [openrouter.ai](https://openrouter.ai/)")
            return

        # Input section
        with st.container():
            st.subheader("📋 User Story Input")

            # Use example if selected
            default_story = st.session_state.get('example_story', '')
            default_context = st.session_state.get('example_context', '')

            user_story = st.text_area(
                "Enter your user story:",
                value=default_story,
                height=150,
                placeholder="Example: As a user, I want to register an account with email and password so that I can access the platform.",
                help="Describe the functionality you want to test"
            )

            additional_context = st.text_area(
                "Additional context (optional):",
                value=default_context,
                height=100,
                placeholder="Example: Email must be valid format. Password minimum 8 characters with numbers and special characters.",
                help="Provide any additional requirements, constraints, or clarifications"
            )

            # Generate button
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                generate_button = st.button(
                    "🚀 Generate Tests",
                    type="primary",
                    use_container_width=True,
                    disabled=not user_story.strip()
                )

            with col2:
                if st.session_state.generation_complete and st.session_state.result:
                    if st.button("📥 Export", use_container_width=True):
                        st.session_state.show_export = True

            with col3:
                if st.session_state.generation_complete:
                    if st.button("🔄 New Generation", use_container_width=True):
                        self.chat.clear_chat()
                        st.session_state.generation_complete = False
                        st.session_state.result = None
                        st.rerun()

        # Generation process
        if generate_button and user_story.strip():
            self.run_generation(user_story, additional_context)

        # Display results if generation is complete
        if st.session_state.generation_complete and st.session_state.result:
            self.display_results()

            # Export section
            if st.session_state.get('show_export', False):
                self.display_export_options()

    def run_generation(self, user_story: str, additional_context: str = ""):
        """
        Run the test generation process

        Args:
            user_story: The user story text
            additional_context: Additional context
        """
        # Clear previous results
        self.chat.clear_chat()
        st.session_state.generation_complete = False
        st.session_state.result = None
        st.session_state.current_iteration_data = []

        # Create message container
        message_container = st.container()

        # Create orchestrator
        try:
            orchestrator = TestGenerationOrchestrator(st.session_state.api_key)

            # Update config
            orchestrator.max_iterations = st.session_state.max_iterations
            orchestrator.coverage_threshold = st.session_state.coverage_threshold

            # Set up callbacks
            def on_message(message: str, message_type: str = "info"):
                """Callback for agent messages"""
                self.chat.add_message(message, role="assistant", message_type=message_type)

            def on_iteration_complete(iteration_data: dict):
                """Callback for iteration completion"""
                st.session_state.current_iteration_data.append(iteration_data)

            orchestrator.set_callbacks(
                on_message=on_message,
                on_iteration_complete=on_iteration_complete
            )

            # Run generation with progress indicator
            with message_container:
                with st.spinner("🤖 AI Agent is generating tests..."):
                    result = orchestrator.run(
                        user_story=user_story,
                        additional_context=additional_context,
                        auto_mode=st.session_state.auto_mode
                    )

                st.session_state.result = result
                st.session_state.generation_complete = True

                # Display chat history
                st.subheader("💬 Generation Log")
                self.chat.display_messages()

        except Exception as e:
            st.error(f"❌ Error during generation: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

    def display_results(self):
        """Display the generation results"""
        if not st.session_state.result:
            return

        result = st.session_state.result

        st.divider()
        st.header("📊 Results")

        # Metrics
        metadata = result['metadata']
        final_validation = metadata.get('final_validation', {})

        self.viz.display_metrics_grid(
            test_count=final_validation.get('test_count', 0),
            coverage=result['coverage'],
            iterations=result['iterations'],
            elapsed_time=metadata.get('elapsed_time', 0)
        )

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📝 Generated Tests",
            "📈 Coverage Analysis",
            "🔄 Iteration History",
            "⚙️ Quality Metrics"
        ])

        with tab1:
            self.display_generated_tests(result)

        with tab2:
            self.display_coverage_analysis(result)

        with tab3:
            self.display_iteration_history(result)

        with tab4:
            self.display_quality_metrics(result)

    def display_generated_tests(self, result: dict):
        """Display the generated test code"""
        st.subheader("Generated Test Code")

        tests = result['tests']

        # Syntax highlighting
        st.code(tests, language='python', line_numbers=True)

        # Download button
        st.download_button(
            label="📄 Download Python File",
            data=tests,
            file_name=f"test_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
            mime="text/x-python"
        )

    def display_coverage_analysis(self, result: dict):
        """Display coverage analysis"""
        st.subheader("Coverage Analysis")

        coverage_analysis = result['metadata'].get('coverage_analysis', {})

        # Coverage gauge
        col1, col2 = st.columns([1, 2])

        with col1:
            fig_gauge = self.viz.create_coverage_gauge(
                result['coverage'],
                st.session_state.coverage_threshold
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            # Scenario breakdown
            scenarios = result['metadata'].get('scenarios', {})
            if scenarios:
                fig_breakdown = self.viz.plot_coverage_breakdown(scenarios)
                st.plotly_chart(fig_breakdown, use_container_width=True)

        # Covered vs Missing scenarios
        st.divider()
        self.viz.display_scenario_coverage(
            covered=coverage_analysis.get('covered_scenarios', []),
            missing=coverage_analysis.get('missing_scenarios', [])
        )

    def display_iteration_history(self, result: dict):
        """Display iteration history"""
        st.subheader("Iteration Progress")

        iteration_history = result.get('iteration_history', [])

        if not iteration_history:
            st.info("No iteration history available")
            return

        # Coverage progress chart
        coverage_scores = [iter_data['coverage'] for iter_data in iteration_history]
        fig_progress = self.viz.plot_coverage_progress(coverage_scores)
        st.plotly_chart(fig_progress, use_container_width=True)

        # Iteration summary
        fig_summary = self.viz.plot_iteration_summary(iteration_history)
        st.plotly_chart(fig_summary, use_container_width=True)

        # Detailed iteration data
        with st.expander("📋 Detailed Iteration Data"):
            for iter_data in iteration_history:
                st.markdown(f"### Iteration {iter_data['iteration']}")
                st.write(f"- Coverage: {iter_data['coverage']:.1%}")
                st.write(f"- Tests: {iter_data['validation']['test_count']}")
                st.write(f"- Covered scenarios: {len(iter_data['covered_scenarios'])}")
                st.write(f"- Missing scenarios: {len(iter_data['missing_scenarios'])}")
                st.divider()

    def display_quality_metrics(self, result: dict):
        """Display quality metrics"""
        st.subheader("Quality Assessment")

        final_validation = result['metadata'].get('final_validation', {})

        # Quality score
        quality_score = final_validation.get('quality_score', 0)
        st.metric("Overall Quality Score", f"{quality_score:.1%}")

        # Validation checks
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Validation Checks")
            checks = {
                "✅ Syntax Valid": final_validation.get('syntax_valid', False),
                "✅ Pytest Compatible": final_validation.get('pytest_compatible', False),
                "✅ Has Assertions": final_validation.get('has_assertions', False),
                "✅ Has Docstrings": final_validation.get('has_docstrings', False),
                "✅ Has Fixtures": final_validation.get('has_fixtures', False),
                "✅ Has Parametrize": final_validation.get('has_parametrize', False)
            }

            for check, passed in checks.items():
                if passed:
                    st.success(check)
                else:
                    st.warning(check.replace('✅', '⚠️'))

        with col2:
            st.markdown("#### Test Statistics")
            st.write(f"- Test Count: {final_validation.get('test_count', 0)}")
            st.write(f"- Assertion Count: N/A")
            st.write(f"- BLEU Score: {final_validation.get('bleu_score', -1):.2f}" if final_validation.get('bleu_score', -1) >= 0 else "- BLEU Score: N/A")

        # Quality issues
        quality_issues = final_validation.get('quality_issues', [])
        recommendations = final_validation.get('recommendations', [])

        self.viz.display_quality_issues(quality_issues, recommendations)

    def display_export_options(self):
        """Display export options"""
        st.divider()
        st.subheader("📤 Export Options")

        if not st.session_state.result:
            st.warning("No results to export")
            return

        result = st.session_state.result
        tests = result['tests']

        col1, col2, col3, col4 = st.columns(4)

        export_metadata = {
            'user_story': "Generated tests",
            'timestamp': datetime.now().isoformat(),
            'coverage': result['coverage'],
            'iterations': result['iterations'],
            'test_count': result['metadata']['final_validation']['test_count']
        }

        with col1:
            if st.button("📄 Export Python", use_container_width=True):
                try:
                    exporter = PythonExporter()
                    path = exporter.export(
                        tests,
                        str(config.EXPORTS_DIR / f"tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"),
                        export_metadata
                    )
                    st.success(f"✅ Exported to {path}")
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")

        with col2:
            if st.button("📊 Export CSV", use_container_width=True):
                try:
                    exporter = CSVExporter()
                    path = exporter.export(
                        tests,
                        str(config.EXPORTS_DIR / f"tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
                        export_metadata
                    )
                    st.success(f"✅ Exported to {path}")
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")

        with col3:
            if st.button("📋 Export JSON", use_container_width=True):
                try:
                    exporter = JSONExporter()
                    path = exporter.export(
                        tests,
                        str(config.EXPORTS_DIR / f"tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
                        export_metadata
                    )
                    st.success(f"✅ Exported to {path}")
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")

        with col4:
            if st.button("📕 Export PDF", use_container_width=True):
                try:
                    exporter = PDFExporter()
                    path = exporter.export(
                        tests,
                        str(config.EXPORTS_DIR / f"tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"),
                        export_metadata
                    )
                    st.success(f"✅ Exported to {path}")
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")

    def run(self):
        """Run the application"""
        self.render_sidebar()
        self.render_main_interface()


# Main entry point
if __name__ == "__main__":
    app = TestGeneratorApp()
    app.run()
