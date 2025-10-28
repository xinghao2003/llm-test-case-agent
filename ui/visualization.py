"""
Visualization components for coverage metrics and progress
"""
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import streamlit as st


class CoverageVisualizer:
    """
    Creates visualizations for test coverage and progress
    """

    def __init__(self):
        """Initialize the visualizer"""
        pass

    def plot_coverage_progress(self, coverage_scores: List[float]) -> go.Figure:
        """
        Plot coverage progress across iterations

        Args:
            coverage_scores: List of coverage scores per iteration

        Returns:
            Plotly figure
        """
        iterations = list(range(1, len(coverage_scores) + 1))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=iterations,
            y=coverage_scores,
            mode='lines+markers',
            name='Coverage',
            line=dict(color='#3498db', width=3),
            marker=dict(size=10, color='#2980b9')
        ))

        fig.update_layout(
            title='Coverage Progress Across Iterations',
            xaxis_title='Iteration',
            yaxis_title='Coverage Score',
            yaxis=dict(range=[0, 1], tickformat='.0%'),
            hovermode='x unified',
            template='plotly_white',
            height=400
        )

        return fig

    def plot_coverage_breakdown(self, scenarios: Dict[str, List[str]]) -> go.Figure:
        """
        Plot breakdown of scenario coverage by category

        Args:
            scenarios: Dict with scenario categories and lists

        Returns:
            Plotly figure
        """
        categories = []
        counts = []

        for category, scenario_list in scenarios.items():
            if scenario_list:
                # Format category name
                formatted_category = category.replace('_', ' ').title()
                categories.append(formatted_category)
                counts.append(len(scenario_list))

        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=counts,
                marker=dict(
                    color=counts,
                    colorscale='Blues',
                    showscale=False
                ),
                text=counts,
                textposition='auto',
            )
        ])

        fig.update_layout(
            title='Test Scenarios by Category',
            xaxis_title='Category',
            yaxis_title='Number of Scenarios',
            template='plotly_white',
            height=400
        )

        return fig

    def plot_iteration_summary(self, iteration_history: List[Dict]) -> go.Figure:
        """
        Plot summary of tests added per iteration

        Args:
            iteration_history: List of iteration data dicts

        Returns:
            Plotly figure
        """
        iterations = []
        test_counts = []
        coverage_values = []

        for iteration in iteration_history:
            iterations.append(f"Iter {iteration['iteration']}")
            # Count tests in this iteration
            test_code = iteration.get('tests', '')
            test_count = iteration.get('validation', {}).get('test_count', 0)
            test_counts.append(test_count)
            coverage_values.append(iteration.get('coverage', 0))

        fig = go.Figure()

        # Add bar chart for test counts
        fig.add_trace(go.Bar(
            x=iterations,
            y=test_counts,
            name='Total Tests',
            marker_color='#3498db',
            yaxis='y1'
        ))

        # Add line chart for coverage
        fig.add_trace(go.Scatter(
            x=iterations,
            y=coverage_values,
            name='Coverage',
            mode='lines+markers',
            marker=dict(size=8, color='#e74c3c'),
            line=dict(color='#e74c3c', width=2),
            yaxis='y2'
        ))

        fig.update_layout(
            title='Tests and Coverage per Iteration',
            xaxis_title='Iteration',
            yaxis=dict(
                title='Number of Tests',
                side='left'
            ),
            yaxis2=dict(
                title='Coverage',
                overlaying='y',
                side='right',
                tickformat='.0%',
                range=[0, 1]
            ),
            template='plotly_white',
            height=400,
            hovermode='x unified'
        )

        return fig

    def create_coverage_gauge(self, coverage: float, threshold: float = 0.8) -> go.Figure:
        """
        Create a gauge chart for coverage

        Args:
            coverage: Current coverage value (0-1)
            threshold: Target threshold

        Returns:
            Plotly figure
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=coverage * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Coverage", 'font': {'size': 24}},
            delta={'reference': threshold * 100, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 100], 'ticksuffix': "%"},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        return fig

    def display_metrics_grid(
        self,
        test_count: int,
        coverage: float,
        iterations: int,
        elapsed_time: float
    ):
        """
        Display metrics in a grid layout

        Args:
            test_count: Number of tests
            coverage: Coverage score
            iterations: Number of iterations
            elapsed_time: Time elapsed in seconds
        """
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Tests Generated", test_count, delta=None)

        with col2:
            st.metric("Coverage", f"{coverage:.1%}", delta=None)

        with col3:
            st.metric("Iterations", iterations, delta=None)

        with col4:
            st.metric("Time", f"{elapsed_time:.1f}s", delta=None)

    def display_scenario_coverage(
        self,
        covered: List[str],
        missing: List[str]
    ):
        """
        Display covered vs missing scenarios

        Args:
            covered: List of covered scenarios
            missing: List of missing scenarios
        """
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Covered Scenarios")
            if covered:
                for scenario in covered:
                    st.write(f"- {scenario}")
            else:
                st.info("No scenarios covered yet")

        with col2:
            st.subheader("❌ Missing Scenarios")
            if missing:
                for scenario in missing:
                    st.write(f"- {scenario}")
            else:
                st.success("All scenarios covered!")

    def display_quality_issues(self, issues: List[str], recommendations: List[str]):
        """
        Display quality issues and recommendations

        Args:
            issues: List of quality issues
            recommendations: List of recommendations
        """
        if issues:
            with st.expander("⚠️ Quality Issues", expanded=False):
                for issue in issues:
                    st.warning(issue)

        if recommendations:
            with st.expander("💡 Recommendations", expanded=False):
                for rec in recommendations:
                    st.info(rec)
