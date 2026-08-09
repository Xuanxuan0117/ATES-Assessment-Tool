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
import hashlib
        

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
            '_navigation_stable': True,
            'case_upload_widget_version': 0,
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

    def _check_mc_results_outdated(self) -> bool:
        """Check if Monte Carlo results are outdated """
        if 'monte_carlo_results' not in st.session_state or st.session_state.monte_carlo_results is None:
            return False
        
        if '_mc_config_hash' not in st.session_state:
            return False
        
        # Build config snapshot with only relevant fields for each distribution type
        distributions_snapshot = {}
        for param_name, dist in st.session_state.get('param_distributions', {}).items():
            dist_type = dist.get('type', 'single_value')
            if dist_type == 'single_value':
                distributions_snapshot[param_name] = {
                    'type': dist_type,
                    'value': dist.get('value')
                }
            elif dist_type == 'range':
                distributions_snapshot[param_name] = {
                    'type': dist_type,
                    'min': dist.get('min'),
                    'max': dist.get('max')
                }
            elif dist_type == 'triangular':
                distributions_snapshot[param_name] = {
                    'type': dist_type,
                    'min': dist.get('min'),
                    'max': dist.get('max'),
                    'most_likely': dist.get('most_likely')
                }
            elif dist_type in ['normal', 'lognormal']:
                distributions_snapshot[param_name] = {
                    'type': dist_type,
                    'mean': dist.get('mean'),
                    'std': dist.get('std')
                }
        
        config_snapshot = {
            'param_distributions': distributions_snapshot,
            'mc_config': {
                'iterations': st.session_state.mc_config.iterations if 'mc_config' in st.session_state else 10000,
                'seed': st.session_state.mc_config.seed if 'mc_config' in st.session_state else None,
                'specify_cooling_flowrate': getattr(st.session_state.mc_config, 'specify_cooling_flowrate', False) if 'mc_config' in st.session_state else False,
                'use_volume_balance': getattr(st.session_state.mc_config, 'use_volume_balance', False) if 'mc_config' in st.session_state else False,
                'constrain_by_thermal_radius': getattr(st.session_state.mc_config, 'constrain_by_thermal_radius', False) if 'mc_config' in st.session_state else False
            }
        }
        config_str = json.dumps(config_snapshot, sort_keys=True, default=str)
        current_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        return current_hash != st.session_state._mc_config_hash
    
    def _check_distribution_params_mismatch(self) -> bool:
        """Check if distribution settings are out of sync with Quick Look parameters"""
        if 'param_distributions' not in st.session_state or 'ates_params' not in st.session_state:
            return False
        
        params = st.session_state.ates_params
        distributions = st.session_state.param_distributions
        
        param_names = [
            'aquifer_temp', 'water_density', 'water_specific_heat_capacity',
            'thermal_recovery_factor',
            'tolerance_in_thermal_recovery', 'heating_number_of_doublets',
            'heating_days', 'cooling_days', 'pump_energy_density',
            'heating_ave_injection_temp', 'heating_temp_to_building',
            'cop_param_a', 'cop_param_b', 'cop_param_c', 'cop_param_d',
            'carbon_intensity', 'cooling_ave_injection_temp', 'cooling_temp_to_building'
        ]
        
        for param_name in param_names:
            if param_name in distributions and hasattr(params, param_name):
                dist = distributions[param_name]
                dist_type = dist.get('type', 'single_value')
                param_value = getattr(params, param_name)
                
                # Get the representative value based on distribution type
                if dist_type == 'single_value':
                    dist_value = dist.get('value')
                elif dist_type == 'triangular':
                    dist_value = dist.get('most_likely')
                elif dist_type in ['normal', 'lognormal']:
                    dist_value = dist.get('mean')
                elif dist_type == 'range':
                    min_val = dist.get('min', 0)
                    max_val = dist.get('max', 0)
                    dist_value = (min_val + max_val) / 2
                else:
                    dist_value = dist.get('value')
                
                if dist_value is not None and param_value is not None:
                    try:
                        if abs(float(dist_value) - float(param_value)) > 1e-6:
                            return True
                    except (TypeError, ValueError):
                        pass
        
        return False
    
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


    def _check_unsaved_parameter_changes(self) -> bool:
        """Check if there are unsaved parameter changes"""
        if 'ates_params' not in st.session_state:
            return False
        
        params = st.session_state.ates_params
        v = st.session_state.get('input_widget_version', 0)
        
        # params name, widget key and temp key
        mappings = [
            ('aquifer_temp', f'aquifer_temp_v{v}', '_temp_aquifer_temp'),
            ('water_density', f'water_density_v{v}', '_temp_water_density'),
            ('water_specific_heat_capacity', f'water_specific_heat_capacity_v{v}', '_temp_water_specific_heat_capacity'),
            ('heating_days', f'heating_days_v{v}', '_temp_heating_days'),
            ('cooling_days', f'cooling_days_v{v}', '_temp_cooling_days'),
            ('heating_temp_to_building', f'heating_temp_to_building_v{v}', '_temp_heating_temp_to_building'),
            ('cooling_temp_to_building', f'cooling_temp_to_building_v{v}', '_temp_cooling_temp_to_building'),
            ('heating_number_of_doublets', f'heating_number_of_doublets_v{v}', '_temp_heating_number_of_doublets'),
            ('heating_ave_injection_temp', f'heating_ave_injection_temp_v{v}', '_temp_heating_ave_injection_temp'),
            ('thermal_recovery_factor', f'thermal_recovery_factor_v{v}', '_temp_thermal_recovery_factor'),
            ('cooling_ave_injection_temp', f'cooling_ave_injection_temp_v{v}', '_temp_cooling_ave_injection_temp'),
            ('cop_param_a', f'cop_param_a_v{v}', '_temp_cop_param_a'),
            ('cop_param_b', f'cop_param_b_v{v}', '_temp_cop_param_b'),
            ('cop_param_c', f'cop_param_c_v{v}', '_temp_cop_param_c'),
            ('cop_param_d', f'cop_param_d_v{v}', '_temp_cop_param_d'),
            ('pump_energy_density', f'pump_energy_density_v{v}', '_temp_pump_energy_density'),
            ('carbon_intensity', f'carbon_intensity_v{v}', '_temp_carbon_intensity'),
        ]
        
        for param_name, widget_key, temp_key in mappings:
            if not hasattr(params, param_name):
                continue
                
            param_value = getattr(params, param_name)
            
            # priortize widget key, then temp key
            if widget_key in st.session_state:
                current_value = st.session_state[widget_key]
            elif temp_key in st.session_state:
                current_value = st.session_state[temp_key]
            else:
                continue
            
            try:
                if abs(float(current_value) - float(param_value)) > 1e-9:
                    return True
            except (TypeError, ValueError):
                if current_value != param_value:
                    return True

        flowrate_choice = st.session_state.get(
            f'specify_flowrate_choice_v{v}',
            'Cool flowrate (compute warm)' if getattr(params, 'specify_cooling_flowrate', False)
            else 'Warm flowrate (compute cool)'
        )
        specify_cooling_flowrate = str(flowrate_choice).startswith('Cool')
        if specify_cooling_flowrate != bool(getattr(params, 'specify_cooling_flowrate', False)):
            return True
        specified_flowrate = st.session_state.get(
            f'specified_flowrate_v{v}',
            st.session_state.get('_temp_specified_flowrate')
        )
        if specified_flowrate is not None:
            reference_flowrate = (
                params.cooling_target_avg_flowrate_pd
                if specify_cooling_flowrate
                else params.heating_target_avg_flowrate_pd
            )
            if abs(float(specified_flowrate) - float(reference_flowrate)) > 1e-9:
                return True

        balance_choice = st.session_state.get(
            f'balance_choice_v{v}',
            'Volume balance' if getattr(params, 'use_volume_balance', False)
            else 'Energy balance'
        )
        use_volume_balance = str(balance_choice).startswith('Volume')
        if use_volume_balance != bool(getattr(params, 'use_volume_balance', False)):
            return True
        specified_tolerance = st.session_state.get(
            f'specified_balance_tolerance_v{v}',
            st.session_state.get('_temp_specified_balance_tolerance')
        )
        if specified_tolerance is not None:
            reference_tolerance = (
                params.tolerance_in_volume_balance
                if use_volume_balance
                else params.tolerance_in_energy_balance
            )
            if abs(float(specified_tolerance) - float(reference_tolerance)) > 1e-9:
                return True
        
        return False
    
    def _render_save_section_stable(self, current_name: str):
        """
        Render save options
        """
        st.sidebar.markdown("**Save Case**")
        
        # Initialize key if not exists
        if "stable_case_name_input" not in st.session_state:
            st.session_state["stable_case_name_input"] = current_name
        
        # Use key only, no value parameter
        new_case_name = st.sidebar.text_input(
            "Case Name",
            key="stable_case_name_input",
            help="Enter a name for your case"
        )
        
        # # Save options
        # save_options = st.sidebar.selectbox(
        #     "Save Type",
        #     ["Parameters Only (Fast)", "Parameters + Results", "Full State (Report)"],
        #     key="stable_save_options",
        #     help="Choose what to save"
        # )
        
        # Check for unsaved parameter changes
        has_unsaved_changes = self._check_unsaved_parameter_changes()
        
        if has_unsaved_changes:
            st.sidebar.warning("You have unsaved parameter changes. Click 'Calculate' first to include them in the save.")
        
        # Check for outdated Monte Carlo results
        mc_outdated = self._check_mc_results_outdated()
        
        if mc_outdated:
            st.sidebar.warning("Monte Carlo results are outdated. Parameters have changed since last run. Please re-run the analysis.")
        
        # Save button
        if st.sidebar.button("Save Case", type="primary", width="stretch", key="stable_save_btn"):
            if has_unsaved_changes:
                st.sidebar.error("Please click 'Calculate' before saving to ensure parameters and results are consistent.")
            else:
                self._save_case_with_name(new_case_name or current_name)
    
    def _render_load_section_stable(self):
        """Render load options - stable version"""
        st.sidebar.markdown("**Load Case**")
        
        # New Case increments this version so a previously selected file cannot be
        # reprocessed after returning to a clean default case.
        upload_widget_version = st.session_state.get('case_upload_widget_version', 0)
        uploaded_file = st.sidebar.file_uploader(
            "Choose case file",
            type=['json'],
            key=f"stable_upload_case_v{upload_widget_version}",
            help="Select a previously saved case file"
        )
        
        # Handle file upload with duplicate processing prevention mechanism
        if uploaded_file is not None:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get('_last_uploaded_file_id') != file_id:
                st.session_state['_last_uploaded_file_id'] = file_id
                self._load_case_with_naming(uploaded_file)
    
    def _render_simplified_quick_actions(self):
        """Render case-level actions."""
        st.sidebar.markdown("**Quick Actions**")

        has_loaded_case = bool(st.session_state.get('_has_case_snapshot', False))
        if st.sidebar.button(
            "Reset Case to Loaded State",
            width="stretch",
            key="reset_loaded_case_btn",
            disabled=not has_loaded_case,
            help="Restore the Quick Look parameters, probability distributions, and Monte Carlo settings from the loaded case."
        ):
            if self.restore_case_snapshot():
                st.sidebar.success("Case restored to its loaded state")
                st.rerun()

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
            'app_state_manager',
            'input_widget_version',
            'case_upload_widget_version',
        }
        
        # Atomic clear all non-core state
        keys_to_clear = [key for key in list(st.session_state.keys()) if key not in core_keys]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        # Force Streamlit to discard the browser-side file selection as well.
        st.session_state['case_upload_widget_version'] = (
            st.session_state.get('case_upload_widget_version', 0) + 1
        )
        
        # Re-initialize to true startup state
        self._initialize_fresh_startup_state()
        
        # Clear reset marker
        st.session_state['_state_initializing'] = False

        st.session_state['_has_case_snapshot'] = False
        if '_case_snapshot_params' in st.session_state:
            del st.session_state['_case_snapshot_params']
        if '_case_snapshot_distributions' in st.session_state:
            del st.session_state['_case_snapshot_distributions']
        if '_case_snapshot_mc_config' in st.session_state:
            del st.session_state['_case_snapshot_mc_config']
        
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
        st.session_state['stable_case_name_input'] = 'Default'
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
        self._sync_mc_config_to_widget_state()
        
        # Navigation stability
        st.session_state['_navigation_stable'] = True
        self._sync_params_to_temp_variables() 
        
        # Ensure not marked as modified after initialization completion
        # Force widget refresh by incrementing version
        st.session_state['input_widget_version'] = st.session_state.get('input_widget_version', 0) + 1
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
            'thermal_recovery_factor',
            'tolerance_in_thermal_recovery', 'heating_number_of_doublets',
            'heating_days', 'cooling_days', 'pump_energy_density',
            'heating_ave_injection_temp', 'heating_temp_to_building',
            'cop_param_a', 'cop_param_b', 'cop_param_c', 'cop_param_d',
            'carbon_intensity', 'cooling_ave_injection_temp', 'cooling_temp_to_building',
            # Feature B - thermal radius parameters
            'screen_length', 'aquifer_porosity', 'rock_specific_heat_capacity',
            'rock_density', 'max_thermal_radius'
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

        flowrate_value = float(params.heating_target_avg_flowrate_pd)
        distributions['borehole_flow_rate'] = {
            'type': 'single_value',
            'value': flowrate_value,
            'min': flowrate_value * 0.8,
            'max': flowrate_value * 1.2,
            'most_likely': flowrate_value,
            'mean': flowrate_value,
            'std': max(flowrate_value * 0.1, 0.01),
            'location': 0.0,
            'use_log_params': False
        }

        balance_value = float(params.tolerance_in_energy_balance)
        distributions['balance_tolerance'] = {
            'type': 'single_value',
            'value': balance_value,
            'min': balance_value * 0.8,
            'max': balance_value * 1.2,
            'most_likely': balance_value,
            'mean': balance_value,
            'std': max(abs(balance_value) * 0.1, 0.01),
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
    
    def _save_case_with_name(self,case_name: str):
        """Save case"""
        try:
            if not case_name or not case_name.strip():
                st.sidebar.error("Case name cannot be empty")
                return
            
            clean_case_name = case_name.strip()
            clean_name = self._clean_filename(clean_case_name)
            
            # # Get data based on save type
            # if "Parameters Only" in save_type:
            #     state_data = self._get_parameters_only()
            #     file_suffix = "params"
            # elif "Parameters + Results" in save_type:
            #     state_data = self._get_parameters_and_results()
            #     file_suffix = "results"
            # else:  # Full State Report
            #     state_data = self._get_full_state()
            #     file_suffix = "report"

            # Get parameters data
            state_data = self._get_parameters_only()
            # Add case metadata
            state_data['case_metadata'] = {
                'case_name': clean_case_name,
                'save_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'ates_tool_version': '8.0'
            }
            
            # Convert to JSON
            state_json = json.dumps(state_data, indent=2, default=str, ensure_ascii=False)
            
            # Generate filename
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"{clean_name}_{timestamp}.json"
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
            st.session_state['stable_case_name_input'] = case_name  
            st.session_state['case_modified'] = False  
            st.session_state['case_last_saved'] = None
            
            # Force widget refresh after loading new data
            st.session_state['input_widget_version'] = st.session_state.get('input_widget_version', 0) + 1
            
            # Re-calculate derived parameters
            if hasattr(st.session_state.ates_params, '__post_init__'):
                st.session_state.ates_params.__post_init__()
            
            self._save_case_snapshot()

            st.sidebar.success(f"Loaded: {case_name}")
            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"Load failed: {str(e)}")
    
    def _execute_atomic_reset_for_load(self):
        """Execute atomic reset for file loading"""
        st.session_state['_state_initializing'] = True
        
       
        core_keys = {'app_state_manager', '_state_initializing', '_last_uploaded_file_id','input_widget_version'}
        
     
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
    
    def _save_case_snapshot(self):
        """
        Save a complete snapshot of the loaded case.
        """
        import copy
        
        # Save parameters snapshot
        if 'ates_params' in st.session_state:
            from tool.core.ates_calculator import ATESParameters
            params = st.session_state.ates_params
            
            # Create a deep copy of parameters
            params_dict = {}
            for field in params.__dataclass_fields__:
                params_dict[field] = getattr(params, field)
            
            st.session_state['_case_snapshot_params'] = params_dict
        
        # Save distributions snapshot
        if 'param_distributions' in st.session_state:
            st.session_state['_case_snapshot_distributions'] = copy.deepcopy(
                st.session_state['param_distributions']
            )

        if 'mc_config' in st.session_state:
            st.session_state['_case_snapshot_mc_config'] = copy.deepcopy(
                st.session_state['mc_config']
            )
        
        # Mark that a snapshot exists
        st.session_state['_has_case_snapshot'] = True

    def restore_case_snapshot(self):
        """
        Restore the complete case to its initial loaded state.
        Returns True if snapshot was restored, False if no snapshot exists
        """
        if not st.session_state.get('_has_case_snapshot', False):
            return False
        
        # Restore parameters
        if '_case_snapshot_params' in st.session_state:
            from tool.core.ates_calculator import ATESParameters
            params = ATESParameters()
            
            for key, value in st.session_state['_case_snapshot_params'].items():
                if hasattr(params, key):
                    setattr(params, key, value)
            
            st.session_state['ates_params'] = params
            
            # Re-sync to temp variables
            self._sync_params_to_temp_variables()
        
        # Restore distributions
        if '_case_snapshot_distributions' in st.session_state:
            import copy
            st.session_state['param_distributions'] = copy.deepcopy(
                st.session_state['_case_snapshot_distributions']
            )
            st.session_state['param_config_version'] = st.session_state.get('param_config_version', 0) + 1
            st.session_state['stable_param_values'] = {}

        if '_case_snapshot_mc_config' in st.session_state:
            import copy
            st.session_state['mc_config'] = copy.deepcopy(
                st.session_state['_case_snapshot_mc_config']
            )
            st.session_state['monte_carlo_iterations'] = st.session_state.mc_config.iterations
            self._sync_mc_config_to_widget_state()
            st.session_state['_mc_operation_widgets_need_sync'] = True
            st.session_state['_mc_settings_widgets_need_sync'] = True

        for key in (
            'results', 'monte_carlo_results', 'sensitivity_results',
            '_mc_config_hash'
        ):
            st.session_state.pop(key, None)
        st.session_state['calculation_count'] = 0
        st.session_state['calculation_status'] = 'not_started'
        st.session_state['last_calculation_time'] = None
        st.session_state['case_modified'] = False

        # Force widget refresh
        st.session_state['input_widget_version'] = st.session_state.get('input_widget_version', 0) + 1
        
        return True

    def restore_quick_look_snapshot(self) -> bool:
        """Restore only the loaded case's Quick Look parameters."""
        if not st.session_state.get('_has_case_snapshot', False):
            return False
        if '_case_snapshot_params' not in st.session_state:
            return False

        from tool.core.ates_calculator import ATESParameters
        params = ATESParameters()
        for key, value in st.session_state['_case_snapshot_params'].items():
            if hasattr(params, key):
                setattr(params, key, value)

        st.session_state['ates_params'] = params
        self._sync_params_to_temp_variables()
        st.session_state['input_widget_version'] = st.session_state.get('input_widget_version', 0) + 1
        return True

    def restore_monte_carlo_snapshot(self) -> bool:
        """Restore only the loaded case's distributions and Monte Carlo settings."""
        if not st.session_state.get('_has_case_snapshot', False):
            return False

        distributions_snapshot = st.session_state.get('_case_snapshot_distributions')
        if distributions_snapshot is not None:
            import copy
            st.session_state['param_distributions'] = copy.deepcopy(distributions_snapshot)
            st.session_state['param_config_version'] = st.session_state.get('param_config_version', 0) + 1
            st.session_state['stable_param_values'] = {}

        mc_config_snapshot = st.session_state.get('_case_snapshot_mc_config')
        if mc_config_snapshot is not None:
            import copy
            st.session_state['mc_config'] = copy.deepcopy(mc_config_snapshot)
            st.session_state['monte_carlo_iterations'] = st.session_state.mc_config.iterations
            self._sync_mc_config_to_widget_state()
            st.session_state['_mc_operation_widgets_need_sync'] = True
            st.session_state['_mc_settings_widgets_need_sync'] = True

        return distributions_snapshot is not None or mc_config_snapshot is not None

    def refresh_case_modified_status(self) -> bool:
        """Recalculate whether the current state differs from the loaded case."""
        if not st.session_state.get('_has_case_snapshot', False):
            return bool(st.session_state.get('case_modified', False))

        params_match = True
        params_snapshot = st.session_state.get('_case_snapshot_params')
        if params_snapshot is not None:
            current_params = st.session_state.get('ates_params')
            params_match = all(
                getattr(current_params, key, object()) == value
                for key, value in params_snapshot.items()
            )

        distributions_match = (
            st.session_state.get('param_distributions')
            == st.session_state.get('_case_snapshot_distributions')
        )
        mc_config_match = (
            '_case_snapshot_mc_config' not in st.session_state
            or st.session_state.get('mc_config') == st.session_state.get('_case_snapshot_mc_config')
        )

        st.session_state['case_modified'] = not (
            params_match and distributions_match and mc_config_match
        )
        return st.session_state['case_modified']


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

    def _sync_monte_carlo_operation_widgets_to_config(self):
        """Capture Screen 2 operation widgets before the sidebar serializes a case."""
        if 'mc_config' not in st.session_state:
            return

        mc_config = st.session_state.mc_config
        flowrate_choice = st.session_state.get('mc_specify_flowrate_choice')
        if flowrate_choice is not None:
            mc_config.specify_cooling_flowrate = str(flowrate_choice).startswith('Cool')

        balance_choice = st.session_state.get('mc_balance_choice')
        if balance_choice is not None:
            mc_config.use_volume_balance = str(balance_choice).startswith('Volume')
        elif 'mc_use_volume_balance' in st.session_state:
            mc_config.use_volume_balance = bool(st.session_state['mc_use_volume_balance'])

        if 'mc_constrain_by_thermal_radius' in st.session_state:
            mc_config.constrain_by_thermal_radius = bool(
                st.session_state['mc_constrain_by_thermal_radius']
            )
    
    def _get_parameters_only(self) -> Dict[str, Any]:
        """Get parameters only data"""
        self._sync_monte_carlo_operation_widgets_to_config()

        data: Dict[str, Any] = {
            'save_type': 'parameters_only',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'version': '8.0'
        }
        
        # Display name mapping 
        display_names = {
            'aquifer_temp': 'Aquifer Temperature (°C)',
            'water_density': 'Water Density (kg/m³)',
            'water_specific_heat_capacity': 'Water Specific Heat Capacity (J/kg/K)',
            'thermal_recovery_factor': 'Thermal Recovery Factor Heating (-)',
            'borehole_flow_rate': 'Borehole Flow Rate (m³/hr)',
            'balance_tolerance': 'Energy or Volume Balance Tolerance (-)',
            'heating_target_avg_flowrate_pd': 'Target Flow Rate Heating (m³/hr)',
            'tolerance_in_energy_balance': 'Energy Balance Tolerance (-)',
            'tolerance_in_thermal_recovery': 'Thermal Recovery Tolerance εRT (-)',
            'use_volume_balance': 'Use Volume Balance',
            'tolerance_in_volume_balance': 'Volume Balance Tolerance εVBR (-)',
            'heating_number_of_doublets': 'Number of Doublets',
            'heating_days': 'Heating Days',
            'cooling_days': 'Cooling Days',
            'pump_energy_density': 'Hydraulic Pump Energy Density (kJ/m³)',
            'heating_ave_injection_temp': 'Cool Well Injection Temperature (°C)',
            'heating_temp_to_building': 'Building Heating Temperature (°C)',
            'cop_param_a': 'COP Parameter A (-)',
            'cop_param_b': 'COP Parameter B (-)',
            'cop_param_c': 'COP Parameter C (-)',
            'cop_param_d': 'COP Parameter D (-)',
            'carbon_intensity': 'Carbon Intensity (gCO₂/kWh)',
            'cooling_ave_injection_temp': 'Warm Well Injection Temperature (°C)',
            'cooling_temp_to_building': 'Building Cooling Temperature (°C)',
            'specify_cooling_flowrate': 'Specify Cooling Flowrate',
            'cooling_target_avg_flowrate_pd': 'Target Flow Rate Cooling (m³/hr)',
            'constrain_by_thermal_radius': 'Constrain by Thermal Radius',
            'screen_length': 'Borehole Screen Length (m)',
            'aquifer_porosity': 'Aquifer Porosity (-)',
            'rock_specific_heat_capacity': 'Rock Specific Heat Capacity (J/kg/°C)',
            'rock_density': 'Rock Density (kg/m³)',
            'max_thermal_radius': 'Maximum Thermal Radius (m)'
        }

        # ATES parameters
        if 'ates_params' in st.session_state:
            params = st.session_state.ates_params
            data['ates_parameters'] = {
                display_names['aquifer_temp']: params.aquifer_temp,
                display_names['water_density']: params.water_density,
                display_names['water_specific_heat_capacity']: params.water_specific_heat_capacity,
                display_names['thermal_recovery_factor']: params.thermal_recovery_factor,
                display_names['heating_target_avg_flowrate_pd']: params.heating_target_avg_flowrate_pd,
                display_names['tolerance_in_energy_balance']: params.tolerance_in_energy_balance,
                display_names['tolerance_in_thermal_recovery']: params.tolerance_in_thermal_recovery,
                display_names['use_volume_balance']: params.use_volume_balance,
                display_names['tolerance_in_volume_balance']: params.tolerance_in_volume_balance,
                display_names['heating_number_of_doublets']: params.heating_number_of_doublets,
                display_names['heating_days']: params.heating_days,
                display_names['cooling_days']: params.cooling_days,
                display_names['pump_energy_density']: params.pump_energy_density,
                display_names['heating_ave_injection_temp']: params.heating_ave_injection_temp,
                display_names['heating_temp_to_building']: params.heating_temp_to_building,
                display_names['cop_param_a']: params.cop_param_a,
                display_names['cop_param_b']: params.cop_param_b,
                display_names['cop_param_c']: params.cop_param_c,
                display_names['cop_param_d']: params.cop_param_d,
                display_names['carbon_intensity']: params.carbon_intensity,
                display_names['cooling_ave_injection_temp']: params.cooling_ave_injection_temp,
                display_names['cooling_temp_to_building']: params.cooling_temp_to_building,
                display_names['specify_cooling_flowrate']: params.specify_cooling_flowrate,
                display_names['cooling_target_avg_flowrate_pd']: params.cooling_target_avg_flowrate_pd,
                display_names['constrain_by_thermal_radius']: params.constrain_by_thermal_radius,
                display_names['screen_length']: params.screen_length,
                display_names['aquifer_porosity']: params.aquifer_porosity,
                display_names['rock_specific_heat_capacity']: params.rock_specific_heat_capacity,
                display_names['rock_density']: params.rock_density,
                display_names['max_thermal_radius']: params.max_thermal_radius
            }
        
        # Probability distributions 
        try:
            param_distributions = getattr(st.session_state, 'param_distributions', {})
            if param_distributions and isinstance(param_distributions, dict):
                # Convert to display names
                converted_distributions = {}
                for key, value in param_distributions.items():
                    display_key = display_names.get(key, key)
                    converted_distributions[display_key] = value
                
                data['param_distributions'] = converted_distributions
        except Exception:
            pass

            # Monte Carlo configuration
        if 'mc_config' in st.session_state:
            mc_config = st.session_state.mc_config
            data['monte_carlo_config'] = {
                'iterations': st.session_state.get('monte_carlo_iterations', mc_config.iterations),
                'seed': mc_config.seed,
                'chunk_size': mc_config.chunk_size,
                'max_workers': mc_config.max_workers,
                'parallel': mc_config.parallel,
                'specify_cooling_flowrate': getattr(mc_config, 'specify_cooling_flowrate', False),
                'use_volume_balance': getattr(mc_config, 'use_volume_balance', False),
                'constrain_by_thermal_radius': getattr(mc_config, 'constrain_by_thermal_radius', False)
            }
        
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
            
            # Reverse mapping 
            reverse_names = {
                'Aquifer Temperature (°C)': 'aquifer_temp',
                'Water Density (kg/m³)': 'water_density',
                'Water Specific Heat Capacity (J/kg/K)': 'water_specific_heat_capacity',
                'Thermal Recovery Factor (-)': 'thermal_recovery_factor',
                'Thermal Recovery Factor Heating (-)': 'thermal_recovery_factor',
                'Thermal Recovery Factor Cooling (-)': 'thermal_recovery_factor_c',
                'Borehole Flow Rate (m³/hr)': 'borehole_flow_rate',
                'Energy or Volume Balance Tolerance (-)': 'balance_tolerance',
                'Thermal Recovery Tolerance εRT (-)': 'tolerance_in_thermal_recovery',
                'Use Volume Balance': 'use_volume_balance',
                'Volume Balance Tolerance εVBR (-)': 'tolerance_in_volume_balance',
                'Target Flow Rate Heating (m³/hr)': 'heating_target_avg_flowrate_pd',
                'Energy Balance Tolerance (-)': 'tolerance_in_energy_balance',
                'Number of Doublets': 'heating_number_of_doublets',
                'Heating Days': 'heating_days',
                'Cooling Days': 'cooling_days',
                'Hydraulic Pump Energy Density (kJ/m³)': 'pump_energy_density',
                'Cool Well Injection Temperature (°C)': 'heating_ave_injection_temp',
                'Building Heating Temperature (°C)': 'heating_temp_to_building',
                'COP Parameter A (-)': 'cop_param_a',
                'COP Parameter B (-)': 'cop_param_b',
                'COP Parameter C (-)': 'cop_param_c',
                'COP Parameter D (-)': 'cop_param_d',
                'Carbon Intensity (gCO₂/kWh)': 'carbon_intensity',
                'Warm Well Injection Temperature (°C)': 'cooling_ave_injection_temp',
                'Building Cooling Temperature (°C)': 'cooling_temp_to_building',
                'Specify Cooling Flowrate': 'specify_cooling_flowrate',
                'Target Flow Rate Cooling (m³/hr)': 'cooling_target_avg_flowrate_pd',
                'Constrain by Thermal Radius': 'constrain_by_thermal_radius',
                'Borehole Screen Length (m)': 'screen_length',
                'Borehole Screen Length Ls (m)': 'screen_length',
                'Aquifer Porosity (-)': 'aquifer_porosity',
                'Aquifer Porosity φ (-)': 'aquifer_porosity',
                'Rock Specific Heat Capacity (J/kg/°C)': 'rock_specific_heat_capacity',
                'Rock Specific Heat Capacity cr (J/kg/°C)': 'rock_specific_heat_capacity',
                'Rock Density (kg/m³)': 'rock_density',
                'Rock Density ρr (kg/m³)': 'rock_density',
                'Maximum Thermal Radius (m)': 'max_thermal_radius',
                'Maximum Thermal Radius r_T,max (m)': 'max_thermal_radius'
            }
            
            params = ATESParameters()
            for key, value in params_dict.items():
                # Try reverse mapping first, then use key directly 
                if key in reverse_names:
                    internal_key = reverse_names[key]
                else:
                    internal_key = key
                if internal_key in ('use_volume_balance', 'specify_cooling_flowrate', 'constrain_by_thermal_radius'):
                    value = bool(value) if not isinstance(value, str) else value.lower() == 'true'
                if hasattr(params, internal_key):
                    setattr(params, internal_key, value)
            
            st.session_state['ates_params'] = params
        
        # Load probability distributions 
        if 'param_distributions' in state_data:
            try:
                loaded_distributions = state_data['param_distributions']
                if isinstance(loaded_distributions, dict):
                    # Reverse mapping (display -> internal)
                    reverse_names = {
                        'Aquifer Temperature (°C)': 'aquifer_temp',
                        'Water Density (kg/m³)': 'water_density',
                        'Water Specific Heat Capacity (J/kg/K)': 'water_specific_heat_capacity',
                        'Thermal Recovery Factor (-)': 'thermal_recovery_factor',
                        'Thermal Recovery Factor Heating (-)': 'thermal_recovery_factor',
                        'Thermal Recovery Factor Cooling (-)': 'thermal_recovery_factor_c',
                        'Borehole Flow Rate (m³/hr)': 'borehole_flow_rate',
                        'Energy or Volume Balance Tolerance (-)': 'balance_tolerance',
                        'Thermal Recovery Tolerance εRT (-)': 'tolerance_in_thermal_recovery',
                        'Use Volume Balance': 'use_volume_balance',
                        'Volume Balance Tolerance εVBR (-)': 'tolerance_in_volume_balance',
                        'Target Flow Rate Heating (m³/hr)': 'heating_target_avg_flowrate_pd',
                        'Energy Balance Tolerance (-)': 'tolerance_in_energy_balance',
                        'Number of Doublets': 'heating_number_of_doublets',
                        'Heating Days': 'heating_days',
                        'Cooling Days': 'cooling_days',
                        'Hydraulic Pump Energy Density (kJ/m³)': 'pump_energy_density',
                        'Cool Well Injection Temperature (°C)': 'heating_ave_injection_temp',
                        'Building Heating Temperature (°C)': 'heating_temp_to_building',
                        'COP Parameter A (-)': 'cop_param_a',
                        'COP Parameter B (-)': 'cop_param_b',
                        'COP Parameter C (-)': 'cop_param_c',
                        'COP Parameter D (-)': 'cop_param_d',
                        'Carbon Intensity (gCO₂/kWh)': 'carbon_intensity',
                        'Warm Well Injection Temperature (°C)': 'cooling_ave_injection_temp',
                        'Building Cooling Temperature (°C)': 'cooling_temp_to_building',
                        'Specify Cooling Flowrate': 'specify_cooling_flowrate',
                        'Target Flow Rate Cooling (m³/hr)': 'cooling_target_avg_flowrate_pd',
                        'Constrain by Thermal Radius': 'constrain_by_thermal_radius',
                        'Borehole Screen Length (m)': 'screen_length',
                        'Borehole Screen Length Ls (m)': 'screen_length',
                        'Aquifer Porosity (-)': 'aquifer_porosity',
                        'Aquifer Porosity φ (-)': 'aquifer_porosity',
                        'Rock Specific Heat Capacity (J/kg/°C)': 'rock_specific_heat_capacity',
                        'Rock Specific Heat Capacity cr (J/kg/°C)': 'rock_specific_heat_capacity',
                        'Rock Density (kg/m³)': 'rock_density',
                        'Rock Density ρr (kg/m³)': 'rock_density',
                        'Maximum Thermal Radius (m)': 'max_thermal_radius',
                        'Maximum Thermal Radius r_T,max (m)': 'max_thermal_radius'
                    }
                    
                    # Convert back to internal names
                    converted_distributions = {}
                    for key, value in loaded_distributions.items():
                        if key in reverse_names:
                            internal_key = reverse_names[key]
                        else:
                            internal_key = key  # backward compatibility
                        converted_distributions[internal_key] = value
                    
                    st.session_state.update({
                        'param_distributions': converted_distributions,
                        'param_config_version': st.session_state.get('param_config_version', 0) + 1,
                        'stable_param_values': {}
                    })
            except Exception:
                self._create_fresh_distributions()
        
        # Load Monte Carlo configuration
        if 'monte_carlo_config' in state_data:
            mc_data = state_data['monte_carlo_config']
            from tool.core.monte_carlo_engine import MonteCarloConfig
            
            st.session_state['monte_carlo_iterations'] = mc_data.get('iterations', 10000)
            
            mc_config = MonteCarloConfig(
                iterations=mc_data.get('iterations', 10000),
                seed=mc_data.get('seed', None),
                chunk_size=mc_data.get('chunk_size', 1000),
                max_workers=mc_data.get('max_workers', None),
                parallel=mc_data.get('parallel', True),
                specify_cooling_flowrate=mc_data.get('specify_cooling_flowrate', False),
                use_volume_balance=mc_data.get('use_volume_balance', False),
                constrain_by_thermal_radius=mc_data.get('constrain_by_thermal_radius', False)
            )
            st.session_state['mc_config'] = mc_config
            self._sync_mc_config_to_widget_state()
            # The upload can be handled after Screen 2 widgets have been created in
            # the current run. Refresh them once at the start of the next run.
            st.session_state['_mc_operation_widgets_need_sync'] = True
            st.session_state['_mc_settings_widgets_need_sync'] = True

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
            ('tolerance_in_thermal_recovery', '_temp_tolerance_in_thermal_recovery'),
            ('use_volume_balance', '_temp_use_volume_balance'),
            ('tolerance_in_volume_balance', '_temp_tolerance_in_volume_balance'),
            ('heating_number_of_doublets', '_temp_heating_number_of_doublets'),
            ('heating_days', '_temp_heating_days'),
            ('cooling_days', '_temp_cooling_days'),
            ('pump_energy_density', '_temp_pump_energy_density'),
            ('heating_ave_injection_temp', '_temp_heating_ave_injection_temp'),
            ('heating_temp_to_building', '_temp_heating_temp_to_building'),
            ('cop_param_a', '_temp_cop_param_a'),
            ('cop_param_b', '_temp_cop_param_b'),
            ('cop_param_c', '_temp_cop_param_c'),
            ('cop_param_d', '_temp_cop_param_d'),
            ('carbon_intensity', '_temp_carbon_intensity'),
            ('cooling_ave_injection_temp', '_temp_cooling_ave_injection_temp'),
            ('cooling_temp_to_building', '_temp_cooling_temp_to_building'),
            # Feature A / B
            ('specify_cooling_flowrate', '_temp_specify_cooling_flowrate'),
            ('cooling_target_avg_flowrate_pd', '_temp_cooling_target_avg_flowrate_pd'),
            ('constrain_by_thermal_radius', '_temp_constrain_by_thermal_radius'),
            ('screen_length', '_temp_screen_length'),
            ('aquifer_porosity', '_temp_aquifer_porosity'),
            ('rock_specific_heat_capacity', '_temp_rock_specific_heat_capacity'),
            ('rock_density', '_temp_rock_density'),
            ('max_thermal_radius', '_temp_max_thermal_radius')
        ]
        
        for param_name, temp_key in temp_mappings:
            if hasattr(params, param_name):
                st.session_state[temp_key] = getattr(params, param_name)

        st.session_state['_temp_specified_flowrate'] = (
            params.cooling_target_avg_flowrate_pd
            if bool(getattr(params, 'specify_cooling_flowrate', False))
            else params.heating_target_avg_flowrate_pd
        )
        st.session_state['_temp_specified_balance_tolerance'] = (
            params.tolerance_in_volume_balance
            if bool(getattr(params, 'use_volume_balance', False))
            else params.tolerance_in_energy_balance
        )

    def _sync_mc_config_to_widget_state(self):
        """Synchronize Monte Carlo config to Screen 2 widget state."""
        if 'mc_config' not in st.session_state:
            return

        mc_config = st.session_state.mc_config
        st.session_state['mc_specify_flowrate_choice'] = (
            "Cool flowrate (compute warm)"
            if bool(getattr(mc_config, 'specify_cooling_flowrate', False))
            else "Warm flowrate (compute cool)"
        )
        st.session_state['mc_use_volume_balance'] = bool(getattr(mc_config, 'use_volume_balance', False))
        st.session_state['mc_balance_choice'] = (
            "Volume balance"
            if bool(getattr(mc_config, 'use_volume_balance', False))
            else "Energy balance"
        )
        st.session_state['mc_constrain_by_thermal_radius'] = bool(getattr(mc_config, 'constrain_by_thermal_radius', False))
    
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
    manager = st.session_state.get('app_state_manager')
    if (
        manager is None
        or not isinstance(manager, ATESAppState)
        or not hasattr(manager, 'restore_quick_look_snapshot')
    ):
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
