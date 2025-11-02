"""
Complete UI integration 
"""

import streamlit as st
import time
from tool.screens.screen1_quick_look import main as screen1_main
from tool.screens.screen2_probabilistic import main as screen2_main
from tool.screens.screen3_distributions import main as screen3_main
from tool.screens.screen4_percentiles import main as screen4_main
from tool.screens.screen5_sensitivity import main as screen5_main
from tool.utils.state_management import get_app_state

st.set_page_config(
    page_title="Imperial ATES Assessment Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_monte_carlo_status():
    # Check completion flag first
    if st.session_state.get('_mc_completed', False):
        return True
    
    mc_results = st.session_state.get('monte_carlo_results')
    if mc_results is not None:
        try:
            # More thorough check
            if hasattr(mc_results, '__len__') and len(mc_results) > 0:
                # Additional validation for DataFrame
                if hasattr(mc_results, 'columns') and len(mc_results.columns) > 0:
                    return True
        except:
            pass
    
    return False

def check_sensitivity_status():
    """Real-time sensitivity status check"""
    sens_results = st.session_state.get('sensitivity_results')
    if sens_results is None:
        return False
    try:
        return isinstance(sens_results, dict) and len(sens_results) > 0
    except:
        return False

def main():
    """Main application with bulletproof navigation and state management"""
    app_state = get_app_state()
 
    render_sidebar(app_state)
    
    # Get current page with fallback
    current_screen = st.session_state.get('current_page', 'Quick Look')
    
    # Handle results page validation
    if current_screen in ['Results - Distributions', 'Results - Percentiles', 'Results - Sensitivity']:
        if not check_monte_carlo_status():
            st.warning("No probabilistic results available. Run Monte Carlo analysis first.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Go to Probabilistic Setup", width="stretch", type="primary", key="goto_prob_main"):
                    st.session_state['current_page'] = 'Probabilistic Setup'
                    st.rerun()
            with col2:
                if st.button("Back to Calculator", width="stretch", key="goto_calc_main"):
                    st.session_state['current_page'] = 'Quick Look'
                    st.rerun()
            return
    
    # Route to screens
    if current_screen == 'Quick Look':
        screen1_main()
    elif current_screen == 'Probabilistic Setup':
        screen2_main()
    elif current_screen == 'Results - Distributions':
        screen3_main()
    elif current_screen == 'Results - Percentiles':
        screen4_main()
    elif current_screen == 'Results - Sensitivity':
        screen5_main()
    
    add_footer()

def render_sidebar(app_state):
    """Render sidebar with bulletproof navigation"""
    st.sidebar.title("Imperial ATES Assessment Tool")
    st.sidebar.markdown("*Imperial Aquifer Thermal Energy Storage System*")
    st.sidebar.markdown("---")
    
    # Simple, reliable navigation
    pages = ['Quick Look', 'Probabilistic Setup', 'Results - Distributions', 'Results - Percentiles', 'Results - Sensitivity']
    page_descriptions = {
        'Quick Look': 'Deterministic calculation',
        'Probabilistic Setup': 'Monte Carlo configuration', 
        'Results - Distributions': 'Frequency distributions',
        'Results - Percentiles': 'Risk analysis',
        'Results - Sensitivity': 'Parameter importance'
    }
    
    current_page = st.session_state.get('current_page', 'Quick Look')
    
    
    selected_page = st.sidebar.selectbox(
        "Navigate to:",
        pages,
        index=pages.index(current_page),
        format_func=lambda x: page_descriptions.get(x, str(x))
    )
    

    if selected_page != current_page:
        st.session_state['current_page'] = selected_page
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Workflow Progress
    render_workflow_progress(app_state)
    
    # Case Management
    app_state.render_case_management()

def render_workflow_progress(app_state):
    """Render workflow progress with real-time status"""
    st.sidebar.subheader("Workflow Progress")
    
    # Real-time status checks
    has_params = st.session_state.get('ates_params') is not None
    has_calculation = st.session_state.get('results') is not None
    has_probabilistic_setup = len([d for d in st.session_state.get('param_distributions', {}).values()
                                  if d.get('type', 'single_value') != 'single_value']) > 0
    has_monte_carlo = check_monte_carlo_status()
    has_sensitivity = check_sensitivity_status()
    
    steps = [
        ("1. Input Parameters", has_params),
        ("2. Deterministic Calc", has_calculation),
        ("3. Probabilistic Setup", has_probabilistic_setup),
        ("4. Monte Carlo", has_monte_carlo),
        ("5. Results Analysis", has_sensitivity)
    ]
    
    for step_name, completed in steps:
        status = "Yes" if completed else "No"
        st.sidebar.markdown(f"**{step_name}:** {status}")
    
    completed_steps = sum(1 for _, completed in steps if completed)
    progress_percentage = (completed_steps / len(steps)) * 100
    
    st.sidebar.progress(progress_percentage / 100)
    st.sidebar.caption(f"Overall Progress: {progress_percentage:.0f}%")

def render_system_status():
    """Render system status with real-time updates"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    
    # Real-time status
    has_deterministic = st.session_state.get('results') is not None
    has_mc = check_monte_carlo_status()
    has_sens = check_sensitivity_status()
    
    st.sidebar.write(f"**Deterministic:** {'Yes' if has_deterministic else 'No'}")
    st.sidebar.write(f"**Monte Carlo:** {'Yes' if has_mc else 'No'}")
    st.sidebar.write(f"**Sensitivity:** {'Yes' if has_sens else 'No'}")
    
    param_distributions = st.session_state.get('param_distributions', {})
    if param_distributions:
        uncertain_count = len([d for d in param_distributions.values() 
                             if d.get('type', 'single_value') != 'single_value'])
        total_params = len(param_distributions)
        
        st.sidebar.write(f"**Parameters:** {total_params} total")
        st.sidebar.write(f"**Uncertain:** {uncertain_count}")

def add_footer():
    """Add application footer"""
    app_state = get_app_state()
    case_name = app_state.get_case_name()
    case_modified = app_state.is_case_modified()
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        modification_status = " (Modified)" if case_modified else ""
        st.markdown(f"**Current Case:** {case_name}{modification_status}")
    
    with col2:
        st.markdown("**Version:** 3.0.0")
    
    with col3:
        st.markdown("**By:** Yixuan Yan")

# Initialize application safely
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = True
    app_state = get_app_state()
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Quick Look'

if __name__ == "__main__":
    main()