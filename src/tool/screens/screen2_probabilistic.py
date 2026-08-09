"""
Screen 2 - Probabilistic Analysis Setup 
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import json
from plotly.subplots import make_subplots
import scipy.stats as stats
from typing import Dict, List, Any, Optional
import time
from dataclasses import replace

from tool.core.ates_calculator import ATESParameters
from tool.core.monte_carlo_engine import ATESMonteCarloEngine, MonteCarloConfig, create_progress_callback
from tool.utils.state_management import get_app_state

BOREHOLE_FLOW_RATE_PARAM = 'borehole_flow_rate'
BALANCE_TOLERANCE_PARAM = 'balance_tolerance'


def _default_distribution(value: float) -> Dict[str, Any]:
    """Create a deterministic distribution configuration from a numeric value."""
    return {
        'type': 'single_value',
        'value': value,
        'min': value * 0.8,
        'max': value * 1.2,
        'most_likely': value,
        'mean': value,
        'std': max(value * 0.1, 0.01),
        'location': 0.0,
        'use_log_params': False,
    }


def _representative_distribution_value(dist: Dict[str, Any]) -> float:
    """Return the value used when a distribution is synced to Quick Look."""
    dist_type = dist.get('type', 'single_value')
    if dist_type == 'single_value':
        return float(dist['value'])
    if dist_type == 'triangular':
        return float(dist['most_likely'])
    if dist_type in ['normal', 'lognormal']:
        return float(dist['mean'])
    return (float(dist['min']) + float(dist['max'])) / 2


def ensure_borehole_flow_rate_distribution() -> None:
    """Migrate legacy warm/cool flow-rate distributions into one shared input."""
    distributions = st.session_state.param_distributions
    legacy_params = ('heating_target_avg_flowrate_pd', 'cooling_target_avg_flowrate_pd')

    if BOREHOLE_FLOW_RATE_PARAM not in distributions:
        source_param = (
            'cooling_target_avg_flowrate_pd'
            if bool(st.session_state.mc_config.specify_cooling_flowrate)
            else 'heating_target_avg_flowrate_pd'
        )
        source = distributions.get(source_param)
        if source is None:
            source_value = getattr(st.session_state.ates_params, source_param)
            source = _default_distribution(float(source_value))
        distributions[BOREHOLE_FLOW_RATE_PARAM] = source.copy()

    migrated = False
    for param_name in legacy_params:
        if param_name in distributions:
            del distributions[param_name]
            migrated = True

    if migrated:
        st.session_state['stable_param_values'] = {}
        st.session_state['param_config_version'] = st.session_state.get('param_config_version', 0) + 1


def ensure_balance_tolerance_distribution() -> None:
    """Migrate legacy energy/volume tolerance distributions into one input."""
    distributions = st.session_state.param_distributions
    legacy_params = ('tolerance_in_energy_balance', 'tolerance_in_volume_balance')

    if BALANCE_TOLERANCE_PARAM not in distributions:
        source_param = (
            'tolerance_in_volume_balance'
            if bool(st.session_state.mc_config.use_volume_balance)
            else 'tolerance_in_energy_balance'
        )
        source = distributions.get(source_param)
        if source is None:
            source_value = getattr(st.session_state.ates_params, source_param)
            source = _default_distribution(float(source_value))
        distributions[BALANCE_TOLERANCE_PARAM] = source.copy()

    migrated = False
    for param_name in legacy_params:
        if param_name in distributions:
            del distributions[param_name]
            migrated = True

    if migrated:
        st.session_state['stable_param_values'] = {}
        st.session_state['param_config_version'] = st.session_state.get('param_config_version', 0) + 1


def initialize_probabilistic_session_state():
    """Initialize session state for probabilistic analysis with robust initialization"""
    if 'ates_params' not in st.session_state:
        st.session_state.ates_params = ATESParameters()
    
    if 'param_distributions' not in st.session_state:
        try:
            st.session_state.param_distributions = initialize_distributions()
        except Exception as e:
            st.warning(f"Distribution initialization warning: {e}")
            st.session_state.param_distributions = {}
            initialize_distributions_from_ates_params()
    if 'monte_carlo_results' not in st.session_state:
        st.session_state.monte_carlo_results = None
    
    if 'monte_carlo_iterations' not in st.session_state:
        st.session_state.monte_carlo_iterations = 10000
    
    if 'sensitivity_results' not in st.session_state:
        st.session_state.sensitivity_results = None
    
    if 'mc_config' not in st.session_state:
        st.session_state.mc_config = MonteCarloConfig()
    ensure_monte_carlo_operation_settings()
    if st.session_state.pop('_mc_operation_widgets_need_sync', False):
        sync_monte_carlo_operation_widget_state()
    if st.session_state.pop('_mc_settings_widgets_need_sync', False):
        sync_monte_carlo_settings_widget_state()
    ensure_borehole_flow_rate_distribution()
    ensure_balance_tolerance_distribution()
    
    if 'param_config_version' not in st.session_state:
        st.session_state.param_config_version = 0
    
    if 'stable_param_values' not in st.session_state:
        st.session_state.stable_param_values = {}

def ensure_distribution_parameter(param_name: str) -> None:
    """Add a missing distribution entry when an existing session predates a new parameter."""
    if param_name in st.session_state.param_distributions:
        return
    if not hasattr(st.session_state.ates_params, param_name):
        return

    current_value = getattr(st.session_state.ates_params, param_name)
    st.session_state.param_distributions[param_name] = _default_distribution(float(current_value))

def ensure_monte_carlo_operation_settings() -> None:
    """Backfill Monte Carlo-only operation settings for older sessions."""
    defaults = {
        'specify_cooling_flowrate': False,
        'use_volume_balance': False,
        'constrain_by_thermal_radius': False,
    }
    for name, value in defaults.items():
        if not hasattr(st.session_state.mc_config, name):
            setattr(st.session_state.mc_config, name, value)

def sync_monte_carlo_operation_widget_state() -> None:
    """Keep Monte Carlo operation widgets aligned after explicit sync/load actions."""
    if 'mc_config' not in st.session_state:
        return

    st.session_state['mc_specify_flowrate_choice'] = (
        "Cool flowrate (compute warm)"
        if bool(st.session_state.mc_config.specify_cooling_flowrate)
        else "Warm flowrate (compute cool)"
    )
    st.session_state['mc_use_volume_balance'] = bool(st.session_state.mc_config.use_volume_balance)
    st.session_state['mc_balance_choice'] = (
        "Volume balance"
        if bool(st.session_state.mc_config.use_volume_balance)
        else "Energy balance"
    )
    st.session_state['mc_constrain_by_thermal_radius'] = bool(st.session_state.mc_config.constrain_by_thermal_radius)


def sync_monte_carlo_settings_widget_state() -> None:
    """Refresh non-operation Monte Carlo widgets after a load or reset."""
    config = st.session_state.mc_config
    st.session_state['mc_iterations_input'] = int(
        st.session_state.get('monte_carlo_iterations', config.iterations)
    )
    st.session_state['mc_seed_input'] = int(config.seed) if config.seed is not None else 0
    st.session_state['mc_parallel_input'] = bool(config.parallel)
    st.session_state['mc_max_workers_input'] = int(config.max_workers or 4)
    st.session_state['mc_chunk_size_input'] = int(config.chunk_size)


def update_monte_carlo_operation_setting(config_name: str, widget_key: str) -> None:
    """Apply a Screen 2 operation change immediately and mark the case modified."""
    if config_name == 'specify_cooling_flowrate':
        new_value = str(st.session_state[widget_key]).startswith('Cool')
    elif config_name == 'use_volume_balance':
        new_value = str(st.session_state[widget_key]).startswith('Volume')
    else:
        new_value = bool(st.session_state[widget_key])

    current_value = bool(getattr(st.session_state.mc_config, config_name))
    if current_value != new_value:
        setattr(st.session_state.mc_config, config_name, new_value)
        from tool.utils.state_management import mark_case_modified
        mark_case_modified()

def initialize_distributions_from_ates_params() -> None:
    """Initialize distributions directly from ATES parameters"""
    params = st.session_state.ates_params
    distributions = {}
    
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
    
    for param_name in probabilistic_params:
        if hasattr(params, param_name):
            current_value = getattr(params, param_name)
            distributions[param_name] = _default_distribution(float(current_value))

    distributions[BOREHOLE_FLOW_RATE_PARAM] = _default_distribution(
        float(params.heating_target_avg_flowrate_pd)
    )
    distributions[BALANCE_TOLERANCE_PARAM] = _default_distribution(
        float(params.tolerance_in_energy_balance)
    )
    
    st.session_state.param_distributions = distributions

def initialize_distributions() -> Dict[str, Dict[str, Any]]:
    """Initialize parameter distribution configurations"""
    params = ATESParameters()
    distributions = {}
    
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
    
    for param_name in probabilistic_params:
        if hasattr(params, param_name):
            current_value = getattr(params, param_name)
            distributions[param_name] = _default_distribution(float(current_value))

    distributions[BOREHOLE_FLOW_RATE_PARAM] = _default_distribution(
        float(params.heating_target_avg_flowrate_pd)
    )
    distributions[BALANCE_TOLERANCE_PARAM] = _default_distribution(
        float(params.tolerance_in_energy_balance)
    )
    
    return distributions

def sync_from_deterministic():
    """Sync parameter values from deterministic calculation to probabilistic setup"""
    for param_name in st.session_state.param_distributions:
        if hasattr(st.session_state.ates_params, param_name):
            current_value = getattr(st.session_state.ates_params, param_name)
            dist = st.session_state.param_distributions[param_name]
            
            dist['value'] = current_value
            dist['mean'] = current_value
            dist['most_likely'] = current_value
            dist['std'] = max(current_value * 0.1, 0.01)

    if 'mc_config' in st.session_state and 'ates_params' in st.session_state:
        st.session_state.mc_config.specify_cooling_flowrate = bool(st.session_state.ates_params.specify_cooling_flowrate)
        st.session_state.mc_config.use_volume_balance = bool(st.session_state.ates_params.use_volume_balance)
        st.session_state.mc_config.constrain_by_thermal_radius = bool(st.session_state.ates_params.constrain_by_thermal_radius)
        sync_monte_carlo_operation_widget_state()

    flowrate_param = (
        'cooling_target_avg_flowrate_pd'
        if bool(st.session_state.ates_params.specify_cooling_flowrate)
        else 'heating_target_avg_flowrate_pd'
    )
    flowrate_value = float(getattr(st.session_state.ates_params, flowrate_param))
    flowrate_dist = st.session_state.param_distributions[BOREHOLE_FLOW_RATE_PARAM]
    flowrate_dist.update({
        'value': flowrate_value,
        'mean': flowrate_value,
        'most_likely': flowrate_value,
        'std': max(flowrate_value * 0.1, 0.01),
    })

    balance_param = (
        'tolerance_in_volume_balance'
        if bool(st.session_state.ates_params.use_volume_balance)
        else 'tolerance_in_energy_balance'
    )
    balance_value = float(getattr(st.session_state.ates_params, balance_param))
    balance_dist = st.session_state.param_distributions[BALANCE_TOLERANCE_PARAM]
    balance_dist.update({
        'value': balance_value,
        'mean': balance_value,
        'most_likely': balance_value,
        'std': max(abs(balance_value) * 0.1, 0.01),
    })

    # Clear stable_param_values cache to force UI refresh
    st.session_state['stable_param_values'] = {}
    st.session_state['param_config_version'] = st.session_state.get('param_config_version', 0) + 1

def sync_to_deterministic():
    """
    Sync parameter values from probabilistic setup to deterministic calculation
    """
    updated_count = 0
    
    for param_name, dist in st.session_state.param_distributions.items():
        if hasattr(st.session_state.ates_params, param_name):
            if dist['type'] == 'single_value':
                new_value = dist['value']
            elif dist['type'] == 'triangular':
                new_value = dist['most_likely']
            elif dist['type'] in ['normal', 'lognormal']:
                new_value = dist['mean']
            else:  # range
                new_value = (dist['min'] + dist['max']) / 2

            if 'number_of_doublets' in param_name:
                new_value = int(round(new_value))

            current_value = getattr(st.session_state.ates_params, param_name)
            if 'number_of_doublets' in param_name:
                has_changed = (current_value != new_value)
            else:
                has_changed = abs(current_value - new_value) > 1e-6
            
            if has_changed:
                setattr(st.session_state.ates_params, param_name, new_value)
                updated_count += 1

    flowrate_value = _representative_distribution_value(
        st.session_state.param_distributions[BOREHOLE_FLOW_RATE_PARAM]
    )
    flowrate_param = (
        'cooling_target_avg_flowrate_pd'
        if bool(st.session_state.mc_config.specify_cooling_flowrate)
        else 'heating_target_avg_flowrate_pd'
    )
    if abs(float(getattr(st.session_state.ates_params, flowrate_param)) - flowrate_value) > 1e-6:
        setattr(st.session_state.ates_params, flowrate_param, flowrate_value)
        updated_count += 1

    balance_value = _representative_distribution_value(
        st.session_state.param_distributions[BALANCE_TOLERANCE_PARAM]
    )
    balance_param = (
        'tolerance_in_volume_balance'
        if bool(st.session_state.mc_config.use_volume_balance)
        else 'tolerance_in_energy_balance'
    )
    if abs(float(getattr(st.session_state.ates_params, balance_param)) - balance_value) > 1e-6:
        setattr(st.session_state.ates_params, balance_param, balance_value)
        updated_count += 1

    operation_params = [
        'specify_cooling_flowrate',
        'use_volume_balance',
        'constrain_by_thermal_radius',
    ]
    for param_name in operation_params:
        if hasattr(st.session_state.mc_config, param_name) and hasattr(st.session_state.ates_params, param_name):
            new_value = bool(getattr(st.session_state.mc_config, param_name))
            if bool(getattr(st.session_state.ates_params, param_name)) != new_value:
                setattr(st.session_state.ates_params, param_name, new_value)
                updated_count += 1

    
    if updated_count > 0:
        st.session_state.ates_params.__post_init__()

        st.session_state['results'] = None
        if '_last_calculation_time' in st.session_state:
            del st.session_state['_last_calculation_time']
        
        return updated_count
    
    return 0

def render_parameter_config(param_name: str, param_label: str):
    """
    Render parameter configuration interface
    """
    dist_config = st.session_state.param_distributions[param_name]
    current_type = dist_config.get('type', 'single_value')
    is_uncertain = current_type != 'single_value'
    version = st.session_state.get('param_config_version', 0)
    
    with st.expander(f"{param_label}", expanded=is_uncertain):
        type_key = f"type_{param_name}_v{version}"
        supported_types = ['single_value', 'range', 'triangular', 'normal', 'lognormal']
        
        new_dist_type = st.selectbox(
            "Parameter Type",
            supported_types,
            index=supported_types.index(current_type),
            key=type_key,
            format_func=lambda x: {
                'single_value': 'Fixed Value (Deterministic)',
                'range': 'Uniform Distribution',
                'triangular': 'Triangular Distribution',
                'normal': 'Normal Distribution',
                'lognormal': 'Log-Normal Distribution'
            }[x]
        )
        
        if new_dist_type != current_type:
            dist_config['type'] = new_dist_type
            from tool.utils.state_management import mark_case_modified
            mark_case_modified()
            st.session_state.param_config_version = version + 1
            st.rerun()
        
        render_distribution_params_stable(param_name, dist_config, new_dist_type, version)
        
        if new_dist_type != 'single_value':
            st.markdown("---")
            render_distribution_preview(param_name, dist_config, param_label)

def render_distribution_params_stable(param_name: str, dist_config: Dict, dist_type: str, version: int):
    """
    Render distribution specific parameters
    """
    stable_key = f"{param_name}_v{version}"
    if stable_key not in st.session_state.stable_param_values:
        st.session_state.stable_param_values[stable_key] = dist_config.copy()
    
    stable_config = st.session_state.stable_param_values[stable_key]
    
    # Check if this is an integer parameter
    is_integer_param = 'number_of_doublets' in param_name
    
    def update_stable_config(key: str, value: Any):
        """
        Callback function to update stable configuration
        """
        if is_integer_param:
            value = int(round(value))
        stable_config[key] = value
        dist_config[key] = value
        from tool.utils.state_management import mark_case_modified
        mark_case_modified()

    def update_stable_config_from_widget(config_key: str, widget_key: str):
        """
        Safely update distribution state from a widget callback.
        New-case resets can briefly invalidate versioned widget keys before
        Streamlit recreates them, so fall back to the current stable value.
        """
        value = st.session_state.get(widget_key, stable_config.get(config_key))
        update_stable_config(config_key, value)

    if dist_type == 'single_value':
        val_key = f"val_{param_name}_v{version}"
        if is_integer_param:
            st.number_input(
                "Value",
                value=int(stable_config.get('value', 0)),
                key=val_key,
                step=1,
                on_change=lambda key=val_key: update_stable_config_from_widget('value', key)
            )
        else:
            st.number_input(
                "Value",
                value=float(stable_config.get('value', 0)),
                key=val_key,
                format="%.4f",
                step=0.0001,
                on_change=lambda key=val_key: update_stable_config_from_widget('value', key)
            )
    
    elif dist_type == 'range':
        col1, col2 = st.columns(2)
        with col1:
            min_key = f"min_{param_name}_v{version}"
            if is_integer_param:
                st.number_input(
                    "Minimum",
                    value=int(stable_config.get('min', 0)),
                    key=min_key,
                    step=1,
                    on_change=lambda key=min_key: update_stable_config_from_widget('min', key)
                )
            else:
                st.number_input(
                    "Minimum",
                    value=float(stable_config.get('min', 0)),
                    key=min_key,
                    format="%.4f",
                    step=0.0001,
                    on_change=lambda key=min_key: update_stable_config_from_widget('min', key)
                )
                
        with col2:
            max_key = f"max_{param_name}_v{version}"
            if is_integer_param:
                st.number_input(
                    "Maximum",
                    value=int(stable_config.get('max', 1)),
                    key=max_key,
                    step=1,
                    on_change=lambda key=max_key: update_stable_config_from_widget('max', key)
                )
            else:
                st.number_input(
                    "Maximum",
                    value=float(stable_config.get('max', 1)),
                    key=max_key,
                    format="%.4f",
                    step=0.0001,
                    on_change=lambda key=max_key: update_stable_config_from_widget('max', key)
                )
        
        if stable_config.get('min', 0) >= stable_config.get('max', 1):
            st.error("Minimum must be less than maximum")
    
    elif dist_type == 'triangular':
        col1, col2, col3 = st.columns(3)
        with col1:
            tri_min_key = f"tri_min_{param_name}_v{version}"
            if is_integer_param:
                st.number_input(
                    "Minimum",
                    value=int(stable_config.get('min', 0)),
                    key=tri_min_key,
                    step=1,
                    on_change=lambda key=tri_min_key: update_stable_config_from_widget('min', key)
                )
            else:
                st.number_input(
                    "Minimum",
                    value=float(stable_config.get('min', 0)),
                    key=tri_min_key,
                    format="%.4f",
                    step=0.0001,
                    on_change=lambda key=tri_min_key: update_stable_config_from_widget('min', key)
                )
                
        with col2:
            tri_ml_key = f"tri_ml_{param_name}_v{version}"
            if is_integer_param:
                st.number_input(
                    "Most Likely",
                    value=int(stable_config.get('most_likely', 1)),
                    key=tri_ml_key,
                    step=1,
                    on_change=lambda key=tri_ml_key: update_stable_config_from_widget('most_likely', key)
                )
            else:
                st.number_input(
                    "Most Likely",
                    value=float(stable_config.get('most_likely', 0.5)),
                    key=tri_ml_key,
                    format="%.4f",
                    step=0.0001,
                    on_change=lambda key=tri_ml_key: update_stable_config_from_widget('most_likely', key)
                )
                
        with col3:
            tri_max_key = f"tri_max_{param_name}_v{version}"
            if is_integer_param:
                st.number_input(
                    "Maximum",
                    value=int(stable_config.get('max', 2)),
                    key=tri_max_key,
                    step=1,
                    on_change=lambda key=tri_max_key: update_stable_config_from_widget('max', key)
                )
            else:
                st.number_input(
                    "Maximum",
                    value=float(stable_config.get('max', 1)),
                    key=tri_max_key,
                    format="%.4f",
                    step=0.0001,
                    on_change=lambda key=tri_max_key: update_stable_config_from_widget('max', key)
                )
        
        min_val = stable_config.get('min', 0)
        ml_val = stable_config.get('most_likely', 0.5)
        max_val = stable_config.get('max', 1)
        if not (min_val <= ml_val <= max_val):
            st.error("Most likely value must be between minimum and maximum")
    
    elif dist_type in ['normal', 'lognormal']:
        col1, col2 = st.columns(2)
        with col1:
            mean_key = f"mean_{param_name}_v{version}"
            if is_integer_param:
                st.number_input(
                    "Mean",
                    value=int(stable_config.get('mean', 0)),
                    key=mean_key,
                    step=1,
                    on_change=lambda key=mean_key: update_stable_config_from_widget('mean', key)
                )
            else:
                st.number_input(
                    "Mean",
                    value=float(stable_config.get('mean', 0)),
                    key=mean_key,
                    format="%.4f",
                    step=0.0001,
                    on_change=lambda key=mean_key: update_stable_config_from_widget('mean', key)
                )
                
        with col2:
            std_key = f"std_{param_name}_v{version}"
            if is_integer_param:
                st.number_input(
                    "Standard Deviation",
                    value=int(stable_config.get('std', 1)),
                    min_value=0,
                    key=std_key,
                    step=1,
                    on_change=lambda key=std_key: update_stable_config_from_widget('std', key)
                )
            else:
                st.number_input(
                    "Standard Deviation",
                    value=float(stable_config.get('std', 0.1)),
                    min_value=0.0,
                    key=std_key,
                    format="%.4f",
                    step=0.0001,
                    on_change=lambda key=std_key: update_stable_config_from_widget('std', key)
                )
        
        if dist_type == 'lognormal':
            col3, col4 = st.columns(2)
            with col3:
                location_key = f"location_{param_name}_v{version}"
                st.number_input(
                    "Location Parameter",
                    value=float(stable_config.get('location', 0.0)),
                    key=location_key,
                    format="%.4f",
                    step=0.0001,
                    help="Minimum possible value (location parameter)",
                    on_change=lambda key=location_key: update_stable_config_from_widget('location', key)
                )
                    
            with col4:
                use_log_key = f"use_log_{param_name}_v{version}"
                st.checkbox(
                    "Use Log Parameters",
                    value=bool(stable_config.get('use_log_params', False)),
                    key=use_log_key,
                    help="Check if mean/std are already in log space",
                    on_change=lambda key=use_log_key: update_stable_config_from_widget('use_log_params', key)
                )
            
            mean_val = stable_config.get('mean', 0)
            location_val = stable_config.get('location', 0.0)
            if mean_val <= location_val:
                st.error("Mean must be greater than location parameter for lognormal distribution")

def render_parameter_groups_tabs():
    """
    Render parameter configuration in organized tabs
    """
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Physical Parameters",           # Section A
        "Demand Parameters",              # Section B
        "System Operation",               # Section C
        "Heat Pump and Carbon Intensity", # Section D
        "Thermal Radius"                  # Section F
    ])

    with tab1:
        render_physical_parameters()

    with tab2:
        render_demand_parameters()

    with tab3:
        render_operational_parameters()

    with tab4:
        render_heatpump_parameters()

    with tab5:
        render_thermal_radius_parameters()

def render_physical_parameters():
    """
    Render physical parameters section (Section A)
    """
    st.subheader("Physical Parameters")
    
    physical_params = [
        'aquifer_temp', 
        'water_density', 
        'water_specific_heat_capacity'
    ]
    
    param_labels = {
        'aquifer_temp': 'Aquifer Temperature (°C)',
        'water_density': 'Water Density (kg/m³)',
        'water_specific_heat_capacity': 'Water Specific Heat Capacity (J/kg/K)'
    }
    
    for param in physical_params:
        if param in st.session_state.param_distributions:
            render_parameter_config(param, param_labels[param])

def render_demand_parameters():
    """
    Render demand parameters section (Section B)
    """
    st.subheader("Demand Parameters")
    
    demand_params = [
        'heating_days',
        'heating_temp_to_building',
        'cooling_days',
        'cooling_temp_to_building'
    ]
    
    param_labels = {
        'heating_days': 'Heating Days',
        'heating_temp_to_building': 'Building Heating Temperature (°C)',
        'cooling_days': 'Cooling Days',
        'cooling_temp_to_building': 'Building Cooling Temperature (°C)'
    }
    
    for param in demand_params:
        if param in st.session_state.param_distributions:
            render_parameter_config(param, param_labels[param])

def render_operational_parameters():
    """
    Render ATES system operational parameters section (Section C)
    """
    st.subheader("ATES System Operation")
    
    render_parameter_config(BOREHOLE_FLOW_RATE_PARAM, 'Borehole Flow Rate (m³/hr)')
    render_parameter_config(BALANCE_TOLERANCE_PARAM, 'Energy or Volume Balance Tolerance (-)')

    operational_params = [
        'heating_number_of_doublets',
        'heating_ave_injection_temp',
        'thermal_recovery_factor',
        'tolerance_in_thermal_recovery',
        'cooling_ave_injection_temp'
    ]
    
    param_labels = {
        'heating_number_of_doublets': 'Number of Doublets (-)',
        'heating_ave_injection_temp': 'Cool well injection temperature (°C)',
        'thermal_recovery_factor': 'Thermal Recovery Factor Heating (-)',
        'tolerance_in_thermal_recovery': 'Thermal Recovery Tolerance εRT (-)',
        'cooling_ave_injection_temp': 'Warm well injection temperature (°C)'
    }
    
    for param in operational_params:
        if param in st.session_state.param_distributions:
            render_parameter_config(param, param_labels[param])

def render_heatpump_parameters():
    """
    Render heat pump and carbon intensity parameters section (Section D)
    """
    st.subheader("Heat Pump and Carbon Intensity")
    
    heatpump_params = [
        'cop_param_a',
        'cop_param_b',
        'cop_param_c',
        'cop_param_d',
        'pump_energy_density',
        'carbon_intensity'
    ]
    
    param_labels = {
        'cop_param_a': 'COP Parameter A (-)',
        'cop_param_b': 'COP Parameter B (-)',
        'cop_param_c': 'COP Parameter C (-)',
        'cop_param_d': 'COP Parameter D (-)',
        'pump_energy_density': 'Pump Energy Density (kJ/m³)',
        'carbon_intensity': 'Carbon Intensity (gCO₂/kWh)'
    }
    
    for param in heatpump_params:
        if param in st.session_state.param_distributions:
            render_parameter_config(param, param_labels[param])


def render_thermal_radius_parameters():
    """
    Render thermal radius parameters section (Feature B, Section F)
    """
    st.subheader("Thermal Radius")

    thermal_params = [
        'screen_length',
        'aquifer_porosity',
        'rock_specific_heat_capacity',
        'rock_density',
        'max_thermal_radius'
    ]

    param_labels = {
        'screen_length': 'Borehole Screen Length (m)',
        'aquifer_porosity': 'Aquifer Porosity (-)',
        'rock_specific_heat_capacity': 'Rock Specific Heat Capacity (J/kg/°C)',
        'rock_density': 'Rock Density (kg/m³)',
        'max_thermal_radius': 'Maximum Thermal Radius (m)'
    }

    for param in thermal_params:
        if param in st.session_state.param_distributions:
            render_parameter_config(param, param_labels[param])


def render_distribution_preview(param_name: str, dist_config: Dict, param_label: str):
    """
    Render a preview of the parameter distribution
    """
    try:
        n_samples = 30000
        rng = np.random.default_rng(42)
        
        if dist_config['type'] == 'single_value':
            samples = np.full(n_samples, dist_config['value'])
        elif dist_config['type'] == 'range':
            samples = rng.uniform(dist_config['min'], dist_config['max'], n_samples)
        elif dist_config['type'] == 'triangular':
            c = (dist_config['most_likely'] - dist_config['min']) / (dist_config['max'] - dist_config['min'])
            samples = stats.triang.rvs(c, loc=dist_config['min'], 
                                     scale=dist_config['max'] - dist_config['min'], 
                                     size=n_samples, random_state=rng)
        elif dist_config['type'] == 'normal':
            samples = rng.normal(dist_config['mean'], dist_config['std'], n_samples)
        elif dist_config['type'] == 'lognormal':
            if dist_config['mean'] > 0:
                mu = np.log(dist_config['mean'])
                sigma = dist_config['std'] / dist_config['mean']
                samples = rng.lognormal(mu, sigma, n_samples)
            else:
                samples = np.full(n_samples, 0)
        else:
            samples = np.full(n_samples, 0)
        
        fig = px.histogram(
            x=samples,
            nbins=100,
            title=f"Distribution Preview: {param_label}",
            labels={'x': param_label, 'y': 'Probability'},
            histnorm='probability'
        )
        
        fig.update_traces(
            marker=dict(
                line=dict(color='black', width=1)
            )
        )
        
        fig.update_layout(
            height=300,
            showlegend=False,
            title={
                'text': f"Distribution Preview: {param_label}",
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            font=dict(color='black'),
            margin=dict(l=60, r=30, t=50, b=50),  # add margin
            xaxis=dict(
                linecolor='black',
                tickcolor='black',
                tickfont=dict(color='black'),
                title_font=dict(color='black'),
                ticks='outside',
                showline=True,
                mirror=True,
                showgrid=False,
            ),
            yaxis=dict(
                linecolor='black',
                tickcolor='black',
                tickfont=dict(color='black'),
                title_font=dict(color='black'),
                ticks='outside',
                showline=True,
                mirror=True,
                automargin=True,
                showgrid=False,
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, width="stretch")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{np.mean(samples):.3f}")
        with col2:
            st.metric("Std Dev", f"{np.std(samples):.3f}")
        with col3:
            st.metric("Min", f"{np.min(samples):.3f}")
        with col4:
            st.metric("Max", f"{np.max(samples):.3f}")
    
    except Exception as e:
        st.error(f"Error generating preview: {str(e)}")

def render_monte_carlo_settings():
    """
    Render Monte Carlo simulation settings
    """
    st.subheader("Monte Carlo Simulation Settings")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        if 'mc_iterations_input' not in st.session_state:
            st.session_state['mc_iterations_input'] = st.session_state.get('monte_carlo_iterations', 10000)
        st.session_state.monte_carlo_iterations = st.number_input(
            "Number of Iterations",
            min_value=1000,
            max_value=100000,
            step=1000,
            help="More iterations = more accurate results but longer computation time",
            key='mc_iterations_input'
        )
    
    with col2:
        if 'mc_seed_input' not in st.session_state:
            st.session_state['mc_seed_input'] = (
                st.session_state.mc_config.seed
                if st.session_state.mc_config.seed is not None else 0
            )
        seed = st.number_input(
            "Random Seed",
            min_value=0,
            help="Set for reproducible results (0 for random)",
            key='mc_seed_input'
        )
        st.session_state.mc_config.seed = seed if seed > 0 else None
    
    with col3:
        if 'mc_parallel_input' not in st.session_state:
            st.session_state['mc_parallel_input'] = st.session_state.mc_config.parallel
        parallel = st.checkbox(
            "Parallel Processing",
            help="Use multiple CPU cores for faster computation",
            key='mc_parallel_input'
        )
        st.session_state.mc_config.parallel = parallel

    st.markdown("### Monte Carlo Operation")
    # `mc_config` is the source of truth for Screen 2 operation choices.  Writing
    # these values before creating the widgets prevents stale widget state from a
    # prior case overriding a just-loaded configuration.
    sync_monte_carlo_operation_widget_state()

    with st.container(border=True):
        flow_col, balance_col, constraint_col = st.columns(3)
        with flow_col:
            st.markdown("**Flowrate basis**")
            st.radio(
                "Flowrate basis",
                options=["Warm flowrate (compute cool)", "Cool flowrate (compute warm)"],
                key="mc_specify_flowrate_choice",
                on_change=lambda: update_monte_carlo_operation_setting(
                    'specify_cooling_flowrate', 'mc_specify_flowrate_choice'
                ),
                help="Select which flow rate is sampled as the specified input for Monte Carlo.",
                label_visibility="collapsed"
            )
            st.session_state.mc_config.specify_cooling_flowrate = st.session_state.mc_specify_flowrate_choice.startswith("Cool")

        with balance_col:
            st.markdown("**Balance basis**")
            st.radio(
                "Balance basis",
                options=["Energy balance", "Volume balance"],
                key="mc_balance_choice",
                on_change=lambda: update_monte_carlo_operation_setting(
                    'use_volume_balance', 'mc_balance_choice'
                ),
                help="Select the balance equation used to calculate the other flow rate.",
                label_visibility="collapsed"
            )

        with constraint_col:
            st.markdown("**Thermal radius**")
            st.session_state.mc_config.constrain_by_thermal_radius = st.checkbox(
                "Apply Thermal Radius Constraint",
                help="If checked, Monte Carlo rejects samples that exceed the maximum thermal radius.",
                key="mc_constrain_by_thermal_radius",
                on_change=lambda: update_monte_carlo_operation_setting(
                    'constrain_by_thermal_radius', 'mc_constrain_by_thermal_radius'
                )
            )
    
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            if 'mc_max_workers_input' not in st.session_state:
                st.session_state['mc_max_workers_input'] = st.session_state.mc_config.max_workers or 4
            max_workers = st.number_input(
                "Max Workers",
                min_value=1,
                max_value=16,
                help="Number of parallel workers (if parallel processing enabled)",
                key='mc_max_workers_input'
            )
            st.session_state.mc_config.max_workers = max_workers
        
        with col2:
            if 'mc_chunk_size_input' not in st.session_state:
                st.session_state['mc_chunk_size_input'] = st.session_state.mc_config.chunk_size
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=100,
                max_value=10000,
                help="Number of iterations per chunk for progress tracking",
                key='mc_chunk_size_input'
            )
            st.session_state.mc_config.chunk_size = chunk_size
    
    st.session_state.mc_config.iterations = st.session_state.monte_carlo_iterations

def render_enabled_parameters_summary():
    """
    Render parameter configuration summary with correct uncertainty counting
    """
    uncertain_params = []
    for name, dist in st.session_state.param_distributions.items():
        if dist['type'] != 'single_value':
            uncertain_params.append(name)
    
    st.subheader("Parameter Configuration Summary")
    
    if not uncertain_params:
        st.info("All parameters use fixed values (deterministic analysis)")
        return
    
    st.success(f"{len(uncertain_params)} parameters configured with uncertainty")
    
    # Display name mapping
    display_names = {
        'aquifer_temp': 'Aquifer Temperature (°C)',
        'water_density': 'Water Density (kg/m³)',
        'water_specific_heat_capacity': 'Water Specific Heat Capacity (J/kg/K)',
        'thermal_recovery_factor': 'Thermal Recovery Factor Heating (-)',
        BOREHOLE_FLOW_RATE_PARAM: 'Borehole Flow Rate (m³/hr)',
        BALANCE_TOLERANCE_PARAM: 'Energy or Volume Balance Tolerance (-)',
        'heating_target_avg_flowrate_pd': 'Target Flow Rate Heating (m³/hr)',
        'cooling_target_avg_flowrate_pd': 'Target Flow Rate Cooling (m³/hr)',
        'tolerance_in_energy_balance': 'Energy Balance Tolerance (-)',
        'tolerance_in_thermal_recovery': 'Thermal Recovery Tolerance εRT (-)',
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
        'cooling_temp_to_building': 'Building Cooling Temperature (°C)'
    }
    
    summary_data = []
    for param in uncertain_params:
        dist = st.session_state.param_distributions[param]
        
        if dist['type'] == 'range':
            range_info = f"Range: [{dist['min']:.3f}, {dist['max']:.3f}]"
        elif dist['type'] == 'triangular':
            range_info = f"Triangular: [{dist['min']:.3f}, {dist['most_likely']:.3f}, {dist['max']:.3f}]"
        elif dist['type'] == 'normal':
            range_info = f"Normal: μ={dist['mean']:.3f}, σ={dist['std']:.3f}"
        elif dist['type'] == 'lognormal':
            range_info = f"Log-Normal: μ={dist['mean']:.3f}, σ={dist['std']:.3f}"
        else:
            range_info = "Unknown"
        
        summary_data.append({
            'Parameter': display_names.get(param, param.replace('_', ' ').title()),
            'Distribution': dist['type'].replace('_', ' ').title(),
            'Range/Parameters': range_info
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, width="stretch", hide_index=True)

def render_monte_carlo_execution():
    """
    Render Monte Carlo execution interface with corrected parameter counting
    """
    uncertain_params = []
    for name, dist in st.session_state.param_distributions.items():
        if dist['type'] != 'single_value':
            uncertain_params.append(name)
    
    # if uncertain_params:
    #     st.info(f"Analysis will vary {len(uncertain_params)} uncertain parameters")
        
    #     st.write("**Parameters with uncertainty:**")
    #     param_list = ", ".join([p.replace('_', ' ').title() for p in uncertain_params])
    #     st.write(param_list)
    # else:
    #     st.info("Deterministic analysis - all parameters use fixed values")
    
    validation_errors = validate_distribution_config()
    if validation_errors:
        st.error("Configuration errors:")
        for error in validation_errors:
            st.error(f"• {error}")
        return
    
    if st.button("Run Analysis", type="primary", width="stretch"):
        run_monte_carlo_analysis()

def validate_distribution_config() -> List[str]:
    """
    Validate distribution configurations
    """
    errors = []
    
    for param_name, dist in st.session_state.param_distributions.items():
        dist_type = dist['type']
        
        if dist_type == 'range':
            if dist['min'] >= dist['max']:
                errors.append(f"{param_name}: Minimum must be less than maximum")
        
        elif dist_type == 'triangular':
            if not (dist['min'] <= dist['most_likely'] <= dist['max']):
                errors.append(f"{param_name}: Most likely value must be between min and max")
        
        elif dist_type in ['normal', 'lognormal']:
            if dist['std'] <= 0:
                errors.append(f"{param_name}: Standard deviation must be positive")
            
            if dist_type == 'lognormal' and dist['mean'] <= 0:
                errors.append(f"{param_name}: Mean must be positive for log-normal distribution")
    
    return errors

def run_monte_carlo_analysis():
    """
    Execute Monte Carlo analysis 
    """
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        mc_base_params = replace(
            st.session_state.ates_params,
            specify_cooling_flowrate=bool(st.session_state.mc_config.specify_cooling_flowrate),
            use_volume_balance=bool(st.session_state.mc_config.use_volume_balance),
            constrain_by_thermal_radius=bool(st.session_state.mc_config.constrain_by_thermal_radius),
        )

        mc_engine = ATESMonteCarloEngine(
            mc_base_params,
            st.session_state.mc_config
        )
        
        progress_callback = create_progress_callback(progress_bar, status_text)
        
        start_time = time.time()
        results_df = mc_engine.run_simulation(
            st.session_state.param_distributions,
            progress_callback
        )
        computation_time = time.time() - start_time
        
        # Store results
        st.session_state.monte_carlo_results = results_df
        st.session_state._last_mc_computation_time = computation_time
        
        # Build config snapshot with only relevant fields for each distribution type
        distributions_snapshot = {}
        for param_name, dist in st.session_state.param_distributions.items():
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
                'iterations': st.session_state.mc_config.iterations,
                'seed': st.session_state.mc_config.seed,
                'specify_cooling_flowrate': st.session_state.mc_config.specify_cooling_flowrate,
                'use_volume_balance': st.session_state.mc_config.use_volume_balance,
                'constrain_by_thermal_radius': st.session_state.mc_config.constrain_by_thermal_radius,
            }
        }
        config_str = json.dumps(config_snapshot, sort_keys=True, default=str)
        st.session_state._mc_config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        # Calculate sensitivity analysis if uncertain parameters exist
        uncertain_params = {name: config for name, config in st.session_state.param_distributions.items() 
                           if config['type'] != 'single_value'}
        
        if uncertain_params:
            rng = np.random.default_rng(st.session_state.mc_config.seed)
            parameter_samples = mc_engine._generate_parameter_samples(
                st.session_state.param_distributions, rng
            )
            sampled_flowrate_param = (
                'cooling_target_avg_flowrate_pd'
                if st.session_state.mc_config.specify_cooling_flowrate
                else 'heating_target_avg_flowrate_pd'
            )
            sampled_balance_param = (
                'tolerance_in_volume_balance'
                if st.session_state.mc_config.use_volume_balance
                else 'tolerance_in_energy_balance'
            )
            parameter_samples = parameter_samples.rename(columns={
                sampled_flowrate_param: BOREHOLE_FLOW_RATE_PARAM,
                sampled_balance_param: BALANCE_TOLERANCE_PARAM,
            })
            
            try:
                sensitivity_results = mc_engine.calculate_sensitivity_analysis(parameter_samples)
                st.session_state.sensitivity_results = sensitivity_results
            except Exception:
                st.session_state.sensitivity_results = None
        else:
            st.session_state.sensitivity_results = None

        # Set completion flag
        st.session_state._mc_completed = True
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Only show success message
        st.success("Monte Carlo analysis completed!")

        st.rerun()

    except Exception as e:
        st.error(f"Monte Carlo analysis failed: {str(e)}")

def display_monte_carlo_results():
   """
   Display stored Monte Carlo results
   """
   results_df = st.session_state.monte_carlo_results
   
   # Calculate basic metrics
   successful_runs = int(results_df['success'].sum()) if 'success' in results_df.columns else len(results_df)
   success_rate = successful_runs / len(results_df) * 100
   
   # Get computation time if stored
   computation_time = st.session_state.get('_last_mc_computation_time', 0)
   
   # Show metrics
   col1, col2, col3, col4 = st.columns(4)
   with col1:
       st.metric("Total Iterations", f"{len(results_df):,}")
   with col2:
       st.metric("Successful", f"{successful_runs:,}")
   with col3:
       st.metric("Success Rate", f"{success_rate:.1f}%")
   with col4:
       st.metric("Computation Time", f"{computation_time:.1f}s")
   
   # Show quick results preview
   if successful_runs > 0:
       st.subheader("Quick Results Preview")
       
       successful_results = results_df[results_df['success'] == True] if 'success' in results_df.columns else results_df
   
       preview_params = [
           'heating_system_cop',
           'cooling_system_cop',
           'energy_balance_ratio',
           'volume_balance_ratio',
           'heating_annual_energy_building_GWhth',
           'cooling_annual_energy_building_GWhth',
           'heating_monthly_to_building',
           'cooling_monthly_to_building',
           'heating_ave_power_to_building_MW',
           'cooling_ave_power_to_building_MW',
           'heating_annual_elec_energy_GWhe',
           'cooling_annual_elec_energy_GWhe',
           'heating_co2_emissions_per_thermal',
           'cooling_co2_emissions_per_thermal'
       ]
       
       param_display_names = {
           'heating_system_cop': 'Heating SCOP',
           'cooling_system_cop': 'Cooling SCOP',
           'energy_balance_ratio': 'Energy Balance Ratio',
           'volume_balance_ratio': 'Volume Balance Ratio',
           'heating_annual_energy_building_GWhth': 'Annual Heating Energy to Building (GWhth)',
           'cooling_annual_energy_building_GWhth': 'Annual Cooling Energy to Building (GWhth)',
           'heating_monthly_to_building': 'Average Monthly Heating to Building (GWhth)',
           'cooling_monthly_to_building': 'Average Monthly Cooling to Building (GWhth)',
           'heating_ave_power_to_building_MW': 'Average Heating Power (MWth)',
           'cooling_ave_power_to_building_MW': 'Average Cooling Power (MWth)',
           'heating_annual_elec_energy_GWhe': 'Annual Electricity for Heating (GWhe)',
           'cooling_annual_elec_energy_GWhe': 'Annual Electricity for Cooling (GWhe)',
           'heating_co2_emissions_per_thermal': 'CO₂ Emissions per Unit Heating (g/kWhth)',
           'cooling_co2_emissions_per_thermal': 'CO₂ Emissions per Unit Cooling (g/kWhth)'
       }
       
       preview_data = []
       for param in preview_params:
           if param in successful_results.columns:
               data = successful_results[param].dropna()
               data = data[np.isfinite(data)] 
               if len(data) > 0:
                   preview_data.append({
                       'Parameter': param_display_names.get(param, param.replace('_', ' ').title()),
                       'Mean': f"{data.mean():.3f}",
                       'Std': f"{data.std():.3f}",
                       'P10': f"{data.quantile(0.10):.3f}",
                       'P50': f"{data.quantile(0.50):.3f}",
                       'P90': f"{data.quantile(0.90):.3f}"
                   })
       
       if preview_data:
           preview_df = pd.DataFrame(preview_data)
           st.dataframe(preview_df, width="stretch", hide_index=True)

def render_monte_carlo_export():
    """
    Render Monte Carlo data export options
    """
    st.markdown("### Raw Monte Carlo Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Full Raw Data", width="stretch"):
            results_csv = st.session_state.monte_carlo_results.to_csv(index=False, encoding='utf-8-sig')
            
            app_state = get_app_state()
            case_name = app_state.get_case_name()
            clean_case_name = app_state._clean_filename(case_name)
            
            st.download_button(
                label="Download Complete Results CSV",
                data=results_csv,
                file_name=f"{clean_case_name}_monte_carlo_raw_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_full_mc_csv",
                width="stretch"
            )
    
    with col2:
        if st.button("Export Key Results Only", width="stretch"):
            
            key_columns = [
                'iteration', 'success',
                'heating_system_cop', 'cooling_system_cop', 'overall_system_cop',
                'heating_annual_energy_building_gwh', 'cooling_annual_energy_building_gwh',
                'total_annual_energy_gwh', 'total_electrical_energy_gwh',
                'heating_co2_emissions', 'cooling_co2_emissions',
                'energy_balance_ratio', 'volume_balance_ratio',
                'heating_direct_mode', 'cooling_direct_mode'
            ]
            
            available_key_columns = [col for col in key_columns 
                                   if col in st.session_state.monte_carlo_results.columns]
            
            key_results_df = st.session_state.monte_carlo_results[available_key_columns]
            key_results_csv = key_results_df.to_csv(index=False, encoding='utf-8-sig')
            
            app_state = get_app_state()
            case_name = app_state.get_case_name()
            clean_case_name = app_state._clean_filename(case_name)
            
            st.download_button(
                label="Download Key Results CSV",
                data=key_results_csv,
                file_name=f"{clean_case_name}_monte_carlo_key_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_key_mc_csv",
                width="stretch"
            )

def reset_monte_carlo_setup() -> bool:
    """Restore Screen 2 to its loaded baseline without changing Quick Look."""
    from tool.utils.state_management import get_app_state, mark_case_modified
    app_state = get_app_state()
    restored_loaded_values = app_state.restore_monte_carlo_snapshot()

    if not restored_loaded_values:
        st.session_state.param_config_version = st.session_state.get('param_config_version', 0) + 1
        st.session_state.stable_param_values = {}
        st.session_state.param_distributions = initialize_distributions()

        st.session_state.mc_config = MonteCarloConfig()
        st.session_state.monte_carlo_iterations = st.session_state.mc_config.iterations
        sync_monte_carlo_operation_widget_state()
        sync_monte_carlo_settings_widget_state()
        st.session_state['_mc_operation_widgets_need_sync'] = True
        st.session_state['_mc_settings_widgets_need_sync'] = True
    
    # Clear all analysis results
    analysis_keys = [
        'monte_carlo_results',
        'sensitivity_results', 
        'calculation_count',
        'last_calculation_time',
        'calculation_status',
        '_mc_config_hash'
    ]
    
    for key in analysis_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state.calculation_count = 0
    st.session_state.calculation_status = 'not_started'
    st.session_state.last_calculation_time = None
    
    if restored_loaded_values:
        app_state.refresh_case_modified_status()
    else:
        mark_case_modified()

    return restored_loaded_values

def main():
    """
    Main function for Screen 2 
    """
    
    # Initialize probabilistic analysis session state variables
    initialize_probabilistic_session_state()
    
    # Check if parameter distributions exist but stable values are not cached
    if st.session_state.get('param_distributions') and not st.session_state.get('stable_param_values'):
        st.session_state.param_config_version = st.session_state.get('param_config_version', 0) + 1  # Increment config version
        st.session_state.stable_param_values = {}  # Initialize empty stable values cache
    
    # Set up page header and description
    st.title("Probabilistic Analysis Setup")
    st.markdown("Configure probability distributions for uncertain parameters and run Monte Carlo analysis")
    
    # Create control buttons for synchronization and reset operations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Button to import parameter values from deterministic calculation
        if st.button("Sync FROM Quick Look", width="stretch",
                     help="Import parameter values from deterministic calculation"):
            sync_from_deterministic()  # Import values from Quick Look screen
            st.session_state.param_config_version = st.session_state.get('param_config_version', 0) + 1  # Update version
            st.session_state.stable_param_values = {}  # Clear cached stable values
            st.success("Synchronized from Quick Look")
            st.rerun()  # Refresh the page to reflect changes
    
    with col2:
        # Button to export representative values to deterministic calculation
        if st.button("Sync TO Quick Look", width="stretch",
                     help="Export representative values to deterministic calculation"):
            sync_to_deterministic()  # Export values to Quick Look screen
            st.success("Synchronized to Quick Look")
            st.rerun()
    
    with col3:
        if st.button("Reset", width="stretch",
                     help="Restore loaded distributions and simulation settings without changing Quick Look"):
            restored_loaded_values = reset_monte_carlo_setup()
            if restored_loaded_values:
                st.success("Monte Carlo setup restored to the loaded case values")
            else:
                st.success("Monte Carlo setup reset to default values")
            st.rerun()  # Refresh the page to reflect changes
    
    st.markdown("---")  # Add visual separator
    
    # Create main tabs for different sections of the interface
    tab1, tab2, tab3 = st.tabs([
        "Parameter Configuration",
        "Simulation Settings", 
        "▶️ Run Analysis"
    ])
    
    with tab1:
        # Tab for configuring probability distributions
        st.markdown("### Configure Probability Distributions")
        render_parameter_groups_tabs()  # Display parameter configuration interface
        st.markdown("---")
        render_enabled_parameters_summary()  # Show summary of enabled uncertain parameters
    
    with tab2:
        # Tab for Monte Carlo simulation settings
        render_monte_carlo_settings()  # Display simulation configuration options
    
    with tab3:
        # Tab for running Monte Carlo analysis
        render_monte_carlo_execution()  # Display execution controls and progress
    
    # Display results if Monte Carlo analysis has been completed
    if st.session_state.monte_carlo_results is not None:
        st.markdown("---")
        display_monte_carlo_results()  # Show analysis results and visualizations
        
        # Provide export options for the results data
        with st.expander("Export Monte Carlo Data", expanded=False):
            render_monte_carlo_export()  # Display data export interface
