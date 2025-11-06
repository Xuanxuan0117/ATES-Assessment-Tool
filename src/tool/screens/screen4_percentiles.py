import streamlit as st
from tool.core.visualization_module import ATESVisualizer

def main():
    st.title("Percentile Analysis")
    
    if st.session_state.get('monte_carlo_results') is None:
        return
    
    visualizer = ATESVisualizer(
        st.session_state.monte_carlo_results, 
        st.session_state.get('sensitivity_results')
    )
    
    visualizer.render_percentile_analysis()