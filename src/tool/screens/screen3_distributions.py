import streamlit as st
from tool.core.visualization_module import ATESVisualizer

def main():
    st.title("Results - Frequency Distributions")
    
    visualizer = ATESVisualizer(
        st.session_state.monte_carlo_results,
        st.session_state.get('sensitivity_results')
    )
    
    visualizer.render_distribution_plots()