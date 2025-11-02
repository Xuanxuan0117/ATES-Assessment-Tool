"""
State Management Utility 
"""

import streamlit as st
from typing import Dict, Any, Optional, Union, cast
import pandas as pd
import numpy as np
import json
import time
import re

class RealTimeStatusChecker:
    """Real-time status checker with no cache delays"""
    
    @staticmethod
    def check_deterministic_results():
        """Check deterministic calculation results"""
        results = st.session_state.get('results')
        return results is not None
    
    @staticmethod 
    def check_monte_carlo_results():
        """Check Monte Carlo results - force real-time check"""
        mc_results = st.session_state.get('monte_carlo_results')
        if mc_results is None:
            return False
        
        # Force check data validity
        try:
            if hasattr(mc_results, '__len__'):
                has_data = len(mc_results) > 0
                # Extra check for successful results
                if has_data and hasattr(mc_results, 'get'):
                    success_col = mc_results.get('success')
                    if success_col is not None:
                        return success_col.sum() > 0
                return has_data
        except:
            return False
    
    @staticmethod
    def check_sensitivity_results():
        """Check sensitivity analysis results - force real-time check"""
        sens_results = st.session_state.get('sensitivity_results')
        if sens_results is None:
            return False
        
        try:
            if isinstance(sens_results, dict):
                return len(sens_results) > 0
            return False
        except:
            return False

class ATESAppState:
    """
    Manages application state for the ATES assessment tool 
    """
    
    def __init__(self):
        """Initialize the application state"""
        self._ensure_session_state()
        self.status_checker = RealTimeStatusChecker()
    
    def _ensure_session_state(self):
        """Ensure all necessary session state variables exist with stable default values"""
        # Use more stable default initialization
        defaults: Dict[str, Any] = {
            'current_page': 'Quick Look',
            'case_name': 'Default',
            'case_modified': False,
            'case_last_saved': None,
            'calculation_count': 0,
            'calculation_status': 'not_started',
            'last_calculation_time': None,
            'param_config_version': 0,
            'stable_param_values': {},
            'monte_carlo_results': None,
            'sensitivity_results': None,
            'results': None,
            'validation_errors': {},
            'monte_carlo_iterations': 10000,
            # Add state management stability markers
            '_state_initializing': False,
            '_last_reset_time': None,
            '_navigation_stable': True
        }
        
        # Only initialize keys that don't exist to avoid overwriting existing state
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
        
        # Ensure ATES parameters and distributions exist
        if 'ates_params' not in st.session_state:
            from tool.core.ates_calculator import ATESParameters
            st.session_state['ates_params'] = ATESParameters()
        
        if 'param_distributions' not in st.session_state:
            self._initialize_default_distributions()
        
        if 'mc_config' not in st.session_state:
            from tool.core.monte_carlo_engine import MonteCarloConfig
            st.session_state['mc_config'] = MonteCarloConfig()
    
    def render_case_management(self):
        """
        Render case management interface 
        """
        st.sidebar.markdown("---")
        st.sidebar.subheader("Case Management")
        
        # Use stable state check to avoid frequent updates
        case_name = st.session_state.get('case_name', 'Default')
        case_modified = st.session_state.get('case_modified', False)
        
        self._render_case_info_stable(case_name, case_modified)
        self._render_save_section_stable(case_name)
        self._render_load_section_stable()
        self._render_simplified_quick_actions()
    
    def _render_case_info_stable(self, case_name: str, case_modified: bool):
        """
        Display case information 
        """
        # Avoid frequent string concatenation and markdown updates
        display_name = f"{case_name}*" if case_modified else case_name
        st.sidebar.markdown(f"**Current Case:** {display_name}")
        
        # Only show when there's a save time
        last_saved = st.session_state.get('case_last_saved')
        if last_saved:
            st.sidebar.caption(f"Last saved: {last_saved}")
    
    def _render_save_section_stable(self, current_name: str):
        """Render save options
        """
        st.sidebar.markdown("**Save Case**")
        
        # Use fixed key to avoid component recreation
        new_case_name = st.sidebar.text_input(
            "Case Name",
            value=current_name,
            key="stable_case_name_input",
            help="Enter a name for your case"
        )
        
        # Only update when truly changed
        if new_case_name != current_name and new_case_name.strip():
            st.session_state['case_name'] = new_case_name.strip()
            self._mark_case_modified_safe()
        
        # Save options
        save_options = st.sidebar.selectbox(
            "Save Type",
            ["Parameters Only (Fast)", "Parameters + Results", "Full State (Report)"],
            key="stable_save_options",
            help="Choose what to save"
        )
        
        # Save button
        if st.sidebar.button("Save Case", type="primary", width="stretch", key="stable_save_btn"):
            self._save_case_with_name(save_options, new_case_name or current_name)
    
    def _render_load_section_stable(self):
        """Render load options - stable version"""
        st.sidebar.markdown("**Load Case**")
        
        # Use stable key
        uploaded_file = st.sidebar.file_uploader(
            "Choose case file",
            type=['json'],
            key="stable_upload_case",
            help="Select a previously saved case file"
        )
        
        # Handle file upload with duplicate processing prevention mechanism
        if uploaded_file is not None:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get('_last_uploaded_file_id') != file_id:
                st.session_state['_last_uploaded_file_id'] = file_id
                self._load_case_with_naming(uploaded_file)
    
    def _render_simplified_quick_actions(self):
        """
        Render simplified quick actions 
        """
        st.sidebar.markdown("**Quick Actions**")
        

        if st.sidebar.button("New Case", width="stretch", key="stable_new_case_btn", 
                    help="Start a completely new case (resets everything to startup state)"):
            self._handle_complete_new_case()
    
    def _handle_complete_new_case(self):
        """
        Handle complete new case creation 
        """
        self._execute_atomic_reset()
    
    def _execute_atomic_reset(self):
        """
        Execute atomic complete reset 
        """
        # Clear confirmation dialog state
        st.session_state.pop('_confirm_new_case_shown', None)
        
        # Set reset marker to prevent intermediate state trigger updates
        st.session_state['_state_initializing'] = True
        st.session_state['_last_reset_time'] = time.time()
        
        # Keep core system components
        core_keys = {
            'app_state_manager'
        }
        
        # Atomic clear all non-core state
        keys_to_clear = [key for key in list(st.session_state.keys()) if key not in core_keys]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Re-initialize to true startup state
        self._initialize_fresh_startup_state()
        
        # Clear reset marker
        st.session_state['_state_initializing'] = False
        
        # Show success message
        st.sidebar.success("New case created - All reset to startup state")
        
        # Force complete reload
        st.rerun()
    
    def _initialize_fresh_startup_state(self):
        """
        Initialize to true startup state 
        """
        # Basic navigation state
        st.session_state['current_page'] = 'Quick Look'
        
        # Case management state - true initial state
        st.session_state['case_name'] = 'Default'
        st.session_state['case_modified'] = False  # Explicitly mark as unmodified
        st.session_state.pop('_confirm_new_case_shown', None)
        st.session_state['case_last_saved'] = None
        
        # Calculation and workflow state 
        st.session_state['calculation_count'] = 0
        st.session_state['calculation_status'] = 'not_started'
        st.session_state['last_calculation_time'] = None
        st.session_state['validation_errors'] = {}
        
        # Result state 
        st.session_state['results'] = None
        st.session_state['monte_carlo_results'] = None
        st.session_state['sensitivity_results'] = None
        
        # Parameter configuration state - reset version control
        st.session_state['param_config_version'] = 0
        st.session_state['stable_param_values'] = {}
        
        # recreate default ATES parameters
        from tool.core.ates_calculator import ATESParameters
        st.session_state['ates_params'] = ATESParameters()
        
        # recreate default probability distribution configuration
        self._create_fresh_distributions()
        
        # recreate Monte Carlo configuration
        from tool.core.monte_carlo_engine import MonteCarloConfig
        st.session_state['monte_carlo_iterations'] = 10000
        st.session_state['mc_config'] = MonteCarloConfig()
        
        # Navigation stability
        st.session_state['_navigation_stable'] = True
        
        # Ensure not marked as modified after initialization completion
        st.session_state['case_modified'] = False
    
    def _create_fresh_distributions(self):
        """Create brand new default probability distributions"""
        if 'ates_params' not in st.session_state:
            from tool.core.ates_calculator import ATESParameters
            st.session_state['ates_params'] = ATESParameters()
            
        params = st.session_state.ates_params
        distributions: Dict[str, Dict[str, Any]] = {}
        
        # All 19 probabilistic parameters
        probabilistic_params = [
            'aquifer_temp', 'water_density', 'water_specific_heat_capacity',
            'thermal_recovery_factor', 'heating_target_avg_flowrate_pd',
            'tolerance_in_energy_balance', 'heating_number_of_doublets',
            'heating_months', 'cooling_months', 'pump_energy_density',
            'heating_ave_injection_temp', 'heating_temp_to_building',
            'cop_param_a', 'cop_param_b', 'cop_param_c', 'cop_param_d',
            'carbon_intensity', 'cooling_ave_injection_temp', 'cooling_temp_to_building'
        ]
        
        # Create default distributions - all parameters are fixed values
        for param_name in probabilistic_params:
            if hasattr(params, param_name):
                current_value = getattr(params, param_name)
                distributions[param_name] = {
                    'type': 'single_value',
                    'value': current_value,
                    'min': current_value * 0.8,
                    'max': current_value * 1.2,
                    'most_likely': current_value,
                    'mean': current_value,
                    'std': max(current_value * 0.1, 0.01),
                    'location': 0.0,
                    'use_log_params': False
                }
        
        st.session_state['param_distributions'] = distributions
    
    def _mark_case_modified_safe(self):
        """mark case as modified """
        if not st.session_state.get('_state_initializing', False):
            st.session_state['case_modified'] = True
    
    def mark_case_modified(self):
        """Mark case as modified """
        self._mark_case_modified_safe()
    
    def _save_case_with_name(self, save_type: str, case_name: str):
        """Save case"""
        try:
            if not case_name or not case_name.strip():
                st.sidebar.error("Case name cannot be empty")
                return
            
            clean_case_name = case_name.strip()
            clean_name = self._clean_filename(clean_case_name)
            
            # Get data based on save type
            if "Parameters Only" in save_type:
                state_data = self._get_parameters_only()
                file_suffix = "params"
            elif "Parameters + Results" in save_type:
                state_data = self._get_parameters_and_results()
                file_suffix = "results"
            else:  # Full State Report
                state_data = self._get_full_state()
                file_suffix = "report"
            
            # Add case metadata
            state_data['case_metadata'] = {
                'case_name': clean_case_name,
                'save_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'save_type': save_type,
                'ates_tool_version': '1.0.0'
            }
            
            # Convert to JSON
            state_json = json.dumps(state_data, indent=2, default=str, ensure_ascii=False)
            
            # Generate filename
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"{clean_name}_{file_suffix}_{timestamp}.json"
            size_kb = len(state_json) // 1024
            
            # Download button
            st.sidebar.download_button(
                label=f"Download {clean_case_name} ({size_kb}KB)",
                data=state_json,
                file_name=filename,
                mime="application/json",
                width="stretch",
                key=f"download_{clean_name}_{timestamp}"
            )
            
            # Update save state
            st.session_state['case_last_saved'] = time.strftime('%H:%M:%S')
            st.session_state['case_modified'] = False
            
            st.sidebar.success(f"{clean_case_name} ready for download")
            
        except Exception as e:
            st.sidebar.error(f"Save failed: {str(e)}")
    
    def _load_case_with_naming(self, uploaded_file):
        """Load case file"""
        try:
            # Read file content
            file_content = uploaded_file.read()
            state_data = json.loads(file_content)
            
            # Extract case name
            case_name = self._extract_case_name(state_data, uploaded_file.name)
            
            # Execute complete reset then load
            self._execute_atomic_reset_for_load()
            self._load_state_data(state_data)
            
            # Set case information
            st.session_state['case_name'] = case_name
            st.session_state['case_modified'] = False  # Loaded file is unmodified
            st.session_state['case_last_saved'] = None
            
            # Re-calculate derived parameters
            if hasattr(st.session_state.ates_params, '__post_init__'):
                st.session_state.ates_params.__post_init__()
            
            st.sidebar.success(f"Loaded: {case_name}")
            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"Load failed: {str(e)}")
    
    def _execute_atomic_reset_for_load(self):
        """Execute atomic reset for file loading"""
        st.session_state['_state_initializing'] = True
        
       
        core_keys = {'app_state_manager', '_state_initializing', '_last_uploaded_file_id'}
        
     
        keys_to_clear = [key for key in list(st.session_state.keys()) if key not in core_keys]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
      
        self._initialize_fresh_startup_state()
    
    def _extract_case_name(self, state_data: Dict[str, Any], filename: str) -> str:
        """Extract case name from state data or filename"""
        if 'case_metadata' in state_data:
            metadata = state_data['case_metadata']
            if 'case_name' in metadata and metadata['case_name'].strip():
                return metadata['case_name']
        
        # Infer from filename
        base_name = filename.replace('.json', '')
        timestamp_pattern = r'_\d{8}_\d{6}$'
        base_name = re.sub(timestamp_pattern, '', base_name)
        
        type_suffixes = ['_params', '_results', '_report', '_full']
        for suffix in type_suffixes:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        
        if not base_name.strip():
            return "Loaded Case"
        
        return base_name.replace('_', ' ').title()
    
    def _clean_filename(self, name: str) -> str:
        """Clean filename"""
        cleaned = re.sub(r'[<>:"/\\|?*]', '', name)
        cleaned = cleaned.replace(' ', '_')
        cleaned = re.sub(r'_+', '_', cleaned)
        cleaned = cleaned.strip('_')
        
        if not cleaned:
            cleaned = "ates_case"
        
        return cleaned
    
    def get_case_name(self) -> str:
        """Get current case name"""
        return st.session_state.get('case_name', 'Default')
    
    def set_case_name(self, name: str):
        """Set case name"""
        if not name or not name.strip():
            name = 'Default'
        st.session_state['case_name'] = name.strip()
        self._mark_case_modified_safe()
    
    def is_case_modified(self) -> bool:
        """Check if case has been modified"""
        return st.session_state.get('case_modified', False)
    
    def _get_parameters_only(self) -> Dict[str, Any]:
        """Get parameters only data"""
        data: Dict[str, Any] = {
            'save_type': 'parameters_only',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'version': '1.0.0'
        }
        
        # ATES parameters
        if 'ates_params' in st.session_state:
            params = st.session_state.ates_params
            data['ates_parameters'] = {
                'aquifer_temp': params.aquifer_temp,
                'water_density': params.water_density,
                'water_specific_heat_capacity': params.water_specific_heat_capacity,
                'thermal_recovery_factor': params.thermal_recovery_factor,
                'heating_target_avg_flowrate_pd': params.heating_target_avg_flowrate_pd,
                'tolerance_in_energy_balance': params.tolerance_in_energy_balance,
                'heating_number_of_doublets': params.heating_number_of_doublets,
                'heating_months': params.heating_months,
                'cooling_months': params.cooling_months,
                'pump_energy_density': params.pump_energy_density,
                'heating_ave_injection_temp': params.heating_ave_injection_temp,
                'heating_temp_to_building': params.heating_temp_to_building,
                'cop_param_a': params.cop_param_a,
                'cop_param_b': params.cop_param_b,
                'cop_param_c': params.cop_param_c,
                'cop_param_d': params.cop_param_d,
                'carbon_intensity': params.carbon_intensity,
                'cooling_ave_injection_temp': params.cooling_ave_injection_temp,
                'cooling_temp_to_building': params.cooling_temp_to_building
            }
        
        # Probability distributions 
        try:
            param_distributions = getattr(st.session_state, 'param_distributions', {})
            if param_distributions and isinstance(param_distributions, dict):
                data['param_distributions'] = param_distributions
        except Exception:
            # type error skip probability distribution saving
            pass
        
        return data
    
    def _get_parameters_and_results(self) -> Dict[str, Any]:
        """Get parameters and results data"""
        data = self._get_parameters_only()
        data['save_type'] = 'parameters_and_results'
        
        # Add deterministic calculation results
        if 'results' in st.session_state and st.session_state.results is not None:
            results = st.session_state.results
            deterministic_results = {}
            
            for attr_name in dir(results):
                if not attr_name.startswith('_'):
                    attr_value = getattr(results, attr_name)
                    if isinstance(attr_value, (int, float, bool)):
                        if attr_value == float('inf'):
                            deterministic_results[attr_name] = 'infinity'
                        elif attr_value == float('-inf'):
                            deterministic_results[attr_name] = '-infinity'
                        else:
                            deterministic_results[attr_name] = attr_value
            
            data['deterministic_results'] = {
                'calculation_results': deterministic_results,
                'calculation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        
        return data
    
    def _get_full_state(self) -> Dict[str, Any]:
        """Get complete state data with comprehensive summary"""
        data = self._get_parameters_and_results()
        data['save_type'] = 'full_state_report'
        
        # Add comprehensive summary if Monte Carlo results exist
        if 'monte_carlo_results' in st.session_state and st.session_state.monte_carlo_results is not None:
            try:
                # Import the exporter class
                from tool.core.visualization_module import ATESResultsExporter
                
                # Create exporter instance
                exporter = ATESResultsExporter(
                    st.session_state.monte_carlo_results,
                    st.session_state.get('sensitivity_results')
                )
                
                # Generate comprehensive report
                comprehensive_report = exporter.generate_comprehensive_report()
                
                if "error" not in comprehensive_report:
                    data['comprehensive_summary'] = comprehensive_report
                    
                    # Add Monte Carlo raw data summary
                    mc_results = st.session_state.monte_carlo_results
                    successful_runs = int(mc_results['success'].sum()) if 'success' in mc_results.columns else len(mc_results)
                    
                    data['monte_carlo_summary'] = {
                        'total_iterations': len(mc_results),
                        'successful_iterations': successful_runs,
                        'success_rate_percent': (successful_runs / len(mc_results) * 100) if len(mc_results) > 0 else 0,
                        'failed_iterations': len(mc_results) - successful_runs,
                        'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Add sensitivity analysis summary if available
                    if st.session_state.get('sensitivity_results'):
                        sens_results = st.session_state.sensitivity_results
                        data['sensitivity_analysis_summary'] = {
                            'output_parameters_analyzed': len(sens_results),
                            'total_correlations_calculated': sum(len(df) for df in sens_results.values()),
                            'available_outputs': list(sens_results.keys())
                        }
                else:
                    # If comprehensive report generation failed, add basic Monte Carlo info
                    mc_results = st.session_state.monte_carlo_results
                    successful_runs = int(mc_results['success'].sum()) if 'success' in mc_results.columns else len(mc_results)
                    
                    data['monte_carlo_summary'] = {
                        'total_iterations': len(mc_results),
                        'successful_iterations': successful_runs,
                        'success_rate_percent': (successful_runs / len(mc_results) * 100) if len(mc_results) > 0 else 0,
                        'error': 'Comprehensive summary generation failed',
                        'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
            
            except Exception as e:
                # If any error occurs, add basic information
                data['monte_carlo_summary'] = {
                    'error': f'Summary generation failed: {str(e)}',
                    'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
        
        return data
    
    def _load_state_data(self, state_data: Dict[str, Any]):
        """Load state data from file"""
        # Load ATES parameters
        if 'ates_parameters' in state_data:
            from tool.core.ates_calculator import ATESParameters
            params_dict = state_data['ates_parameters']
            
            params = ATESParameters()
            for key, value in params_dict.items():
                if hasattr(params, key):
                    setattr(params, key, value)
            
            st.session_state['ates_params'] = params
        
        # Load probability distributions 
        if 'param_distributions' in state_data:
            try:
                loaded_distributions = state_data['param_distributions']
                if isinstance(loaded_distributions, dict):
                    st.session_state.update({
                        'param_distributions': loaded_distributions,
                        'param_config_version': st.session_state.get('param_config_version', 0) + 1,
                        'stable_param_values': {}
                    })
            except Exception:
                # If there's a type error, use default distributions
                self._create_fresh_distributions()
        
        
        from tool.core.ates_calculator import ATESParameters
        ATESParameters.enable_validation()
        
      
        if 'ates_params' in st.session_state:
            self._sync_params_to_temp_variables()
        
        # Ensure state is marked as unmodified
        st.session_state['_state_initializing'] = False

    def _sync_params_to_temp_variables(self):
        """Synchronize parameters to temporary variables"""
        params = st.session_state.ates_params
        temp_mappings = [
            ('aquifer_temp', '_temp_aquifer_temp'),
            ('water_density', '_temp_water_density'),
            ('water_specific_heat_capacity', '_temp_water_specific_heat_capacity'),
            ('thermal_recovery_factor', '_temp_thermal_recovery_factor'),
            ('heating_target_avg_flowrate_pd', '_temp_heating_target_avg_flowrate_pd'),
            ('tolerance_in_energy_balance', '_temp_tolerance_in_energy_balance'),
            ('heating_number_of_doublets', '_temp_heating_number_of_doublets'),
            ('heating_months', '_temp_heating_months'),
            ('cooling_months', '_temp_cooling_months'),
            ('pump_energy_density', '_temp_pump_energy_density'),
            ('heating_ave_injection_temp', '_temp_heating_ave_injection_temp'),
            ('heating_temp_to_building', '_temp_heating_temp_to_building'),
            ('cop_param_a', '_temp_cop_param_a'),
            ('cop_param_b', '_temp_cop_param_b'),
            ('cop_param_c', '_temp_cop_param_c'),
            ('cop_param_d', '_temp_cop_param_d'),
            ('carbon_intensity', '_temp_carbon_intensity'),
            ('cooling_ave_injection_temp', '_temp_cooling_ave_injection_temp'),
            ('cooling_temp_to_building', '_temp_cooling_temp_to_building')
        ]
        
        for param_name, temp_key in temp_mappings:
            if hasattr(params, param_name):
                st.session_state[temp_key] = getattr(params, param_name)
    
    def _initialize_default_distributions(self):
        """Initialize default distributions"""
        self._create_fresh_distributions()
    
    def render_system_status(self):
        """Render system status """
        st.sidebar.markdown("---")
        st.sidebar.subheader("System Status")
        
        # Real-time status check to avoid caching delays
        has_deterministic = self.status_checker.check_deterministic_results()
        has_mc_results = self.status_checker.check_monte_carlo_results()
        has_sens_results = self.status_checker.check_sensitivity_results()
        
        # Use more stable status display
        st.sidebar.write(f"**Deterministic:** {'Yes' if has_deterministic else 'No'}")
        st.sidebar.write(f"**Monte Carlo:** {'Yes' if has_mc_results else 'No'}")
        st.sidebar.write(f"**Sensitivity:** {'Yes' if has_sens_results else 'No'}")
        
        # Parameter statistics
        param_distributions = getattr(st.session_state, 'param_distributions', {})
        if param_distributions:
            uncertain_count = len([d for d in param_distributions.values() 
                                 if d.get('type', 'single_value') != 'single_value'])
            total_params = len(param_distributions)
            
            st.sidebar.write(f"**Parameters:** {total_params} total")
            st.sidebar.write(f"**Uncertain:** {uncertain_count}")
        
        # Configuration version (debug info)
        config_version = st.session_state.get('param_config_version', 0)
        if config_version > 0:
            st.sidebar.caption(f"Config v{config_version}")
    
    def has_monte_carlo_results(self) -> bool:
        """Check if Monte Carlo results exist"""
        return bool(self.status_checker.check_monte_carlo_results())
    
    def has_sensitivity_results(self) -> bool:
        """Check if sensitivity analysis results exist"""
        return self.status_checker.check_sensitivity_results()


def get_app_state() -> ATESAppState:
    """Get or create application state manager singleton"""
    if 'app_state_manager' not in st.session_state:
        st.session_state['app_state_manager'] = ATESAppState()
    return cast(ATESAppState, st.session_state['app_state_manager'])


def mark_case_modified():
    """Mark current case as modified"""
    app_state = get_app_state()
    app_state.mark_case_modified()


def track_parameter_change(param_name: str, old_value: Any, new_value: Any):
    """Track parameter changes and mark case as modified when values differ"""
    if old_value != new_value:
        mark_case_modified()


def validate_parameter_range(value: float, min_val: float, max_val: float, param_name: str) -> str:
    """Validate parameter is within acceptable range"""
    try:
        if value < min_val or value > max_val:
            return f"{param_name} must be between {min_val} and {max_val}"
        return ""
    except (TypeError, ValueError):
        return f"{param_name} must be a valid number"


def check_calculation_dependencies() -> bool:
    """Check if all calculation dependencies are satisfied"""
    if 'ates_params' not in st.session_state:
        return False
    
    if st.session_state.get('current_page') == 'Probabilistic Setup':
        uncertain_params = sum(1 for dist in st.session_state.get('param_distributions', {}).values() 
                              if dist.get('type', 'single_value') != 'single_value')
        return uncertain_params > 0
    
    return True


def reset_application_state() -> None:
    """Reset entire application state to clean default state"""
    app_state = get_app_state()
    app_state._execute_atomic_reset()


def format_parameter_value(value: Any, param_type: str = 'float', decimal_places: int = 3) -> str:
    """Format parameter value for consistent display in interface"""
    if value is None:
        return "N/A"
    
    try:
        if param_type == 'float':
            return f"{float(value):.{decimal_places}f}"
        elif param_type == 'int':
            return f"{int(value)}"
        elif param_type == 'percentage':
            return f"{float(value) * 100:.1f}%"
        else:
            return str(value)
    except (ValueError, TypeError):
        return str(value)


def get_parameter_summary() -> Dict[str, Any]:
    """Generate parameter summary for system diagnostics"""
    app_state = get_app_state()
    
    return {
        'input_parameters_count': len(st.session_state.get('input_parameters', {})),
        'probabilistic_parameters_count': len(st.session_state.get('probabilistic_parameters', {})),
        'has_results': app_state.has_monte_carlo_results(),
        'has_sensitivity': app_state.has_sensitivity_results(),
        'ready_for_calculation': check_calculation_dependencies(),
        'case_name': app_state.get_case_name(),
        'case_modified': app_state.is_case_modified()
    }