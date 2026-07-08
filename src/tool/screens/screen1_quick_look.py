"""
Screen 1 - Quick Calculation 
"""
import streamlit as st
import pandas as pd
import time
import json
from dataclasses import fields

# Import calculator and utilities
from tool.core.ates_calculator import ATESParameters, ATESCalculator
from tool.utils.state_management import get_app_state, mark_case_modified

# SESSION STATE MANAGEMENT

def initialize_session_state():
    """
    Initialize session state with case management
    """
    if 'ates_params' not in st.session_state:
        st.session_state.ates_params = ATESParameters()
    
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    if 'calculation_count' not in st.session_state:
        st.session_state.calculation_count = 0
    
    # Initialize case management state
    app_state = get_app_state()

def update_all_parameters_from_temp():
    """
    Update all parameters from temporary variables to ates_params
    """
    # Define parameter mapping between UI temp variables and actual parameters
    temp_params = [
        # A. Basic Physical Parameters
        ('aquifer_temp', '_temp_aquifer_temp'),
        ('water_density', '_temp_water_density'),
        ('water_specific_heat_capacity', '_temp_water_specific_heat_capacity'),
        ('thermal_recovery_factor', '_temp_thermal_recovery_factor'),
        ('tolerance_in_thermal_recovery', '_temp_tolerance_in_thermal_recovery'),
        ('use_volume_balance', '_temp_use_volume_balance'),
        ('tolerance_in_volume_balance', '_temp_tolerance_in_volume_balance'),
        
        # B. System Operational Parameters
        ('specify_cooling_flowrate', '_temp_specify_cooling_flowrate'),
        ('heating_target_avg_flowrate_pd', '_temp_heating_target_avg_flowrate_pd'),
        ('cooling_target_avg_flowrate_pd', '_temp_cooling_target_avg_flowrate_pd'),
        ('tolerance_in_energy_balance', '_temp_tolerance_in_energy_balance'),
        ('heating_number_of_doublets', '_temp_heating_number_of_doublets'),
        ('heating_days', '_temp_heating_days'),
        ('cooling_days', '_temp_cooling_days'),
        ('pump_energy_density', '_temp_pump_energy_density'),
        ('heating_ave_injection_temp', '_temp_heating_ave_injection_temp'),
        ('heating_temp_to_building', '_temp_heating_temp_to_building'),
        
        # C. COP Parameters
        ('cop_param_a', '_temp_cop_param_a'),
        ('cop_param_b', '_temp_cop_param_b'),
        ('cop_param_c', '_temp_cop_param_c'),
        ('cop_param_d', '_temp_cop_param_d'),
        ('carbon_intensity', '_temp_carbon_intensity'),
        
        # D. Cooling Side Parameters
        ('cooling_ave_injection_temp', '_temp_cooling_ave_injection_temp'),
        ('cooling_temp_to_building', '_temp_cooling_temp_to_building'),

        # F. Thermal radius parameters (Feature B)
        ('constrain_by_thermal_radius', '_temp_constrain_by_thermal_radius'),
        ('screen_length', '_temp_screen_length'),
        ('aquifer_porosity', '_temp_aquifer_porosity'),
        ('rock_specific_heat_capacity', '_temp_rock_specific_heat_capacity'),
        ('rock_density', '_temp_rock_density'),
        ('max_thermal_radius', '_temp_max_thermal_radius'),
    ]
    
    # Check if any parameter has changed
    has_changes = False
    # Iterate through all parameter mappings to compare and update values
    for param_name, temp_key in temp_params:
        if temp_key in st.session_state:
            new_value = st.session_state[temp_key]
            old_value = getattr(st.session_state.ates_params, param_name)
            
            # # For integer parameters, compare directly; for float parameters, use threshold to avoid precision issues
            if param_name == 'heating_number_of_doublets':
                if old_value != new_value:
                    setattr(st.session_state.ates_params, param_name, new_value)
                    has_changes = True
            elif param_name in ('use_volume_balance', 'specify_cooling_flowrate', 'constrain_by_thermal_radius'):
                if old_value != new_value:
                    setattr(st.session_state.ates_params, param_name, bool(new_value))
                    has_changes = True
            else:
                if abs(old_value - new_value) > 1e-9:
                    setattr(st.session_state.ates_params, param_name, new_value)
                    has_changes = True
    
    # If changes detected, recalculate derived parameters and mark case as modified
    if has_changes:
        st.session_state.ates_params.__post_init__()
        mark_case_modified()
        
        # Quick Look and Probabilistic settings are intentionally independent.
        # Use the explicit Sync FROM/TO Quick Look buttons on Screen 2 to copy values.

def sync_param_to_distribution(param_name: str, value: float):
    """
    Sync parameter value to probabilistic distribution settings
    """
    if 'param_distributions' not in st.session_state:
        return
    
    if param_name in st.session_state.param_distributions:
        dist = st.session_state.param_distributions[param_name]
        old_value = dist.get('value', None)
        
        # only update if value actually changed
        if old_value != value:
            dist['value'] = value
            dist['mean'] = value
            dist['most_likely'] = value
            # dist['min'] = value * 0.8
            # dist['max'] = value * 1.2
            # dist['std'] = max(value * 0.1, 0.01)

        # Clear stable_param_values cache to force UI refresh
            st.session_state['stable_param_values'] = {}
            st.session_state['param_config_version'] = st.session_state.get('param_config_version', 0) + 1

def sync_all_params_to_distributions():
    """
    Synchronize all parameters to probability distribution settings
    """
    if 'param_distributions' not in st.session_state:
        return
    
    param_names = [
        'aquifer_temp', 'water_density', 'water_specific_heat_capacity', 'thermal_recovery_factor',
        'heating_target_avg_flowrate_pd', 'tolerance_in_energy_balance', 'heating_number_of_doublets',
        'heating_days', 'cooling_days', 'pump_energy_density',
        'heating_ave_injection_temp', 'heating_temp_to_building',
        'cop_param_a', 'cop_param_b', 'cop_param_c', 'cop_param_d', 'carbon_intensity',
        'cooling_ave_injection_temp', 'cooling_temp_to_building',
        'screen_length', 'aquifer_porosity', 'rock_specific_heat_capacity',
        'rock_density', 'max_thermal_radius'
    ]

    for param_name in param_names:
        if param_name in st.session_state.param_distributions:
            current_value = getattr(st.session_state.ates_params, param_name)
            sync_param_to_distribution(param_name, current_value)


def initialize_default_distributions():
    """
    Initialize default distributions, if not exists called after calculation
    """
    if 'param_distributions' not in st.session_state:
        params = st.session_state.ates_params
        distributions = {}
        
        probabilistic_params = [
            'aquifer_temp', 'water_density', 'water_specific_heat_capacity', 'thermal_recovery_factor',
            'heating_target_avg_flowrate_pd', 'tolerance_in_energy_balance', 'heating_number_of_doublets',
            'heating_days', 'cooling_days', 'pump_energy_density', 
            'heating_ave_injection_temp', 'heating_temp_to_building',
            'cop_param_a', 'cop_param_b', 'cop_param_c', 'cop_param_d',
            'carbon_intensity', 'cooling_ave_injection_temp', 'cooling_temp_to_building',
            'screen_length', 'aquifer_porosity', 'rock_specific_heat_capacity',
            'rock_density', 'max_thermal_radius'
        ]

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
        
        st.session_state.param_distributions = distributions
    # else:
    #     # Update existing distributions with current parameter values
    #     sync_all_params_to_distributions()

# Parameter input sections

def calculate_flowrate_preview(**overrides):
    """
    Preview Feature A dependent flowrate without mutating session state or distributions.
    """
    base_params = st.session_state.ates_params
    param_values = {
        field.name: getattr(base_params, field.name)
        for field in fields(ATESParameters)
        if field.init and hasattr(base_params, field.name)
    }
    param_values.update(overrides)
    preview_params = ATESParameters(**param_values)
    if preview_params.specify_cooling_flowrate:
        return preview_params.heating_target_avg_flowrate_pd
    return preview_params.cooling_target_avg_flowrate_pd


def render_parameter_section_a():
    """
    Physical Parameters (3 parameters)
    """
    v = st.session_state.get('input_widget_version', 0)
    
    with st.expander("A. Physical Parameters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            aquifer_temp = st.number_input(
                "Aquifer Temperature (°C)",
                value=float(st.session_state.ates_params.aquifer_temp),
                min_value=0.01,
                step=0.1,
                format="%.2f",
                help="Aquifer temperature",
                key=f"aquifer_temp_v{v}"
            )
            
            water_density = st.number_input(
                "Water Density (kg/m³)",
                value=float(st.session_state.ates_params.water_density),
                min_value=0.01,
                step=0.1,
                format="%.2f",
                help="Length of the cooling season",
                key=f"water_density_v{v}"
            )
        
        with col2:
            water_specific_heat_capacity = st.number_input(
                "Water Specific Heat Capacity (J/kg/K)",
                value=float(st.session_state.ates_params.water_specific_heat_capacity),
                min_value=0.01,
                step=1.0,
                format="%.2f",
                help="Water specific heat capacity",
                key=f"water_specific_heat_capacity_v{v}"
            )
        
        st.session_state['_temp_aquifer_temp'] = aquifer_temp
        st.session_state['_temp_water_density'] = water_density
        st.session_state['_temp_water_specific_heat_capacity'] = water_specific_heat_capacity


def render_parameter_section_b():
    """
    Demand Parameters (4 parameters) 
    """
    v = st.session_state.get('input_widget_version', 0)
    
    with st.expander("B. Demand Parameters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            heating_days = st.number_input(
                "Heating Days",
                value=float(st.session_state.ates_params.heating_days),
                min_value=0.0,
                max_value=365.0,
                step=1.0,
                format="%.2f",
                help="Length of the heating",
                key=f"heating_days_v{v}"
            )
            
            heating_temp_to_building = st.number_input(
                "Building Heating Temperature (°C)",
                value=float(st.session_state.ates_params.heating_temp_to_building),
                min_value=0.0,
                step=1.0,
                format="%.2f",
                help="Building heating temperature",
                key=f"heating_temp_to_building_v{v}"
            )
        
        with col2:
            cooling_days = st.number_input(
                "Cooling Days",
                value=float(st.session_state.ates_params.cooling_days),
                min_value=0.0,
                max_value=365.0,
                step=1.0,
                format="%.2f",
                help="Length of the cooling",
                key=f"cooling_days_v{v}"
            )
            
            cooling_temp_to_building = st.number_input(
                "Building Cooling Temperature (°C)",
                value=float(st.session_state.ates_params.cooling_temp_to_building),
                min_value=0.0,
                step=0.1,
                format="%.2f",
                help="Building cooling temperature",
                key=f"cooling_temp_to_building_v{v}"
            )
        
        st.session_state['_temp_heating_days'] = heating_days
        st.session_state['_temp_cooling_days'] = cooling_days
        st.session_state['_temp_heating_temp_to_building'] = heating_temp_to_building
        st.session_state['_temp_cooling_temp_to_building'] = cooling_temp_to_building


def render_parameter_section_c():
    """
    ATES System Operation (6 parameters) 
    """
    v = st.session_state.get('input_widget_version', 0)
    
    with st.expander("C. System Operation", expanded=False):
        # Feature A: choose which flowrate the user specifies (the other is computed + capped)
        specify_choice = st.radio(
            "Specify flowrate",
            options=["Warm flowrate (compute cool)", "Cool flowrate (compute warm)"],
            index=1 if bool(st.session_state.ates_params.specify_cooling_flowrate) else 0,
            help="The specified rate is treated as the maximum; the computed rate magnitude is capped to it.",
            key=f"specify_flowrate_choice_v{v}",
            horizontal=True,
        )
        specify_cooling_flowrate = specify_choice.startswith("Cool")

        flow_col, recovery_col = st.columns(2)

        with flow_col:
            if not specify_cooling_flowrate:
                heating_target_avg_flowrate_pd = st.number_input(
                    "Target Flow Rate Heating (m³/hr)",
                    value=float(st.session_state.ates_params.heating_target_avg_flowrate_pd),
                    min_value=0.01,
                    step=1.0,
                    format="%.2f",
                    help="Target (warm) flow rate per doublet for heating",
                    key=f"heating_target_avg_flowrate_pd_v{v}"
                )
                cooling_target_avg_flowrate_pd = float(st.session_state.ates_params.cooling_target_avg_flowrate_pd)
            else:
                cooling_target_avg_flowrate_pd = st.number_input(
                    "Target Flow Rate Cooling (m³/hr)",
                    value=float(st.session_state.ates_params.cooling_target_avg_flowrate_pd) or 60.0,
                    min_value=0.01,
                    step=1.0,
                    format="%.2f",
                    help="Target (cool) flow rate per doublet for cooling",
                    key=f"cooling_target_avg_flowrate_pd_v{v}"
                )
                heating_target_avg_flowrate_pd = float(st.session_state.ates_params.heating_target_avg_flowrate_pd)

        with recovery_col:
            thermal_recovery_factor = st.number_input(
                "Thermal Recovery Factor (-)",
                value=float(st.session_state.ates_params.thermal_recovery_factor),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.2f",
                help="Thermal recovery efficiency",
                key=f"thermal_recovery_factor_v{v}"
            )

        flowrate_preview_slot = st.empty()

        col1, col2 = st.columns(2)

        with col1:
            heating_number_of_doublets = st.number_input(
                "Number of Doublets",
                value=int(st.session_state.ates_params.heating_number_of_doublets),
                min_value=1,
                step=1,
                help="Number of well doublets",
                key=f"heating_number_of_doublets_v{v}"
            )
            
            heating_ave_injection_temp = st.number_input(
                "Cool well injection temperature (°C)",
                value=float(st.session_state.ates_params.heating_ave_injection_temp),
                min_value=0.0,
                step=0.1,
                format="%.2f",
                help="Cool well injection temperature (< Aquifer Temperature)",
                key=f"heating_ave_injection_temp_v{v}"
            )
            if heating_ave_injection_temp >= st.session_state.ates_params.aquifer_temp:
                st.warning("Cool well injection temperature must be < aquifer temperature")
        
        with col2:
            tolerance_in_energy_balance = st.number_input(
                "Energy Balance Tolerance (-)",
                value=float(st.session_state.ates_params.tolerance_in_energy_balance),
                step=0.01,
                format="%.2f",
                help="Energy balance tolerance",
                key=f"tolerance_in_energy_balance_v{v}"
            )

            cooling_ave_injection_temp = st.number_input(
                "Warm well injection temperature (°C)",
                value=float(st.session_state.ates_params.cooling_ave_injection_temp),
                min_value=0.0,
                step=0.1,
                format="%.2f",
                help="Warm well injection temperature (> Aquifer Temperature)",
                key=f"cooling_ave_injection_temp_v{v}"
            )
            if cooling_ave_injection_temp <= st.session_state.ates_params.aquifer_temp:
                st.warning("Warm well injection temperature must be > aquifer temperature")

        _, volume_toggle_col = st.columns(2)
        with volume_toggle_col:
            use_volume_balance = st.checkbox(
                "Use Volume Balance for Cooling Flow Rate",
                value=bool(st.session_state.ates_params.use_volume_balance),
                help="If checked, uses volume balance; otherwise uses energy balance",
                key=f"use_volume_balance_v{v}"
            )

        col3, col4 = st.columns(2)

        with col3:
            tolerance_in_thermal_recovery = st.number_input(
                "Thermal Recovery Tolerance εRT (-)",
                value=float(st.session_state.ates_params.tolerance_in_thermal_recovery),
                min_value=-1.0,
                max_value=1.0,
                step=0.01,
                format="%.3f",
                help="Difference between heating and cooling thermal recovery factors",
                key=f"tolerance_in_thermal_recovery_v{v}"
            )

        with col4:
            if use_volume_balance:
                tolerance_in_volume_balance = st.number_input(
                    "Volume Balance Tolerance εVBR (-)",
                    value=float(st.session_state.ates_params.tolerance_in_volume_balance),
                    step=0.01,
                    format="%.3f",
                    key=f"tolerance_in_volume_balance_v{v}"
                )
            else:
                tolerance_in_volume_balance = st.session_state.ates_params.tolerance_in_volume_balance

        try:
            preview_flowrate = calculate_flowrate_preview(
                aquifer_temp=st.session_state.get('_temp_aquifer_temp', st.session_state.ates_params.aquifer_temp),
                thermal_recovery_factor=thermal_recovery_factor,
                tolerance_in_thermal_recovery=tolerance_in_thermal_recovery,
                use_volume_balance=use_volume_balance,
                tolerance_in_volume_balance=tolerance_in_volume_balance,
                specify_cooling_flowrate=specify_cooling_flowrate,
                heating_target_avg_flowrate_pd=heating_target_avg_flowrate_pd,
                cooling_target_avg_flowrate_pd=cooling_target_avg_flowrate_pd,
                tolerance_in_energy_balance=tolerance_in_energy_balance,
                heating_days=st.session_state.get('_temp_heating_days', st.session_state.ates_params.heating_days),
                cooling_days=st.session_state.get('_temp_cooling_days', st.session_state.ates_params.cooling_days),
                heating_ave_injection_temp=heating_ave_injection_temp,
                cooling_ave_injection_temp=cooling_ave_injection_temp,
            )
            if specify_cooling_flowrate:
                flowrate_preview_slot.caption(f"Computed warm flow rate: {preview_flowrate:.2f} m³/hr (capped)")
            else:
                flowrate_preview_slot.caption(f"Computed cool flow rate: {preview_flowrate:.2f} m³/hr (capped)")
        except Exception:
            flowrate_preview_slot.caption("Computed flow rate preview unavailable for the current inputs.")

        st.session_state['_temp_specify_cooling_flowrate'] = specify_cooling_flowrate
        st.session_state['_temp_heating_target_avg_flowrate_pd'] = heating_target_avg_flowrate_pd
        st.session_state['_temp_cooling_target_avg_flowrate_pd'] = cooling_target_avg_flowrate_pd
        st.session_state['_temp_heating_number_of_doublets'] = heating_number_of_doublets
        st.session_state['_temp_thermal_recovery_factor'] = thermal_recovery_factor
        st.session_state['_temp_tolerance_in_energy_balance'] = tolerance_in_energy_balance
        st.session_state['_temp_heating_ave_injection_temp'] = heating_ave_injection_temp
        st.session_state['_temp_cooling_ave_injection_temp'] = cooling_ave_injection_temp
        st.session_state['_temp_tolerance_in_thermal_recovery'] = tolerance_in_thermal_recovery
        st.session_state['_temp_use_volume_balance'] = use_volume_balance
        st.session_state['_temp_tolerance_in_volume_balance'] = tolerance_in_volume_balance

def render_parameter_section_d():
    """
    Heat Pump and Carbon Intensity (6 parameters) 
    """
    v = st.session_state.get('input_widget_version', 0)
    
    with st.expander("D. Heat Pump and Carbon Intensity", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            cop_param_a = st.number_input(
                "COP Parameter A (-)",
                value=float(st.session_state.ates_params.cop_param_a),
                min_value=0.0,
                step=1.0,
                format="%.2f",
                help="COP model parameter A",
                key=f"cop_param_a_v{v}"
            )
            
            cop_param_c = st.number_input(
                "COP Parameter C (-)",
                value=float(st.session_state.ates_params.cop_param_c),
                step=0.01,
                format="%.2f",
                help="COP model parameter C",
                key=f"cop_param_c_v{v}"
            )
            
            pump_energy_density = st.number_input(
                "Hydraulic Pump Energy Density (kJ/m³)",
                value=float(st.session_state.ates_params.pump_energy_density),
                min_value=0.0,
                step=10.0,
                format="%.2f",
                help="Hydraulic pump energy density",
                key=f"pump_energy_density_v{v}"
            )
        
        with col2:
            cop_param_b = st.number_input(
                "COP Parameter B (-)",
                value=float(st.session_state.ates_params.cop_param_b),
                min_value=0.01,
                step=0.1,
                format="%.2f",
                help="COP model parameter B (must be positive)",
                key=f"cop_param_b_v{v}"
            )
            
            cop_param_d = st.number_input(
                "COP Parameter D (-)",
                value=float(st.session_state.ates_params.cop_param_d),
                min_value=0.0,
                step=0.1,
                format="%.2f",
                help="COP model parameter D",
                key=f"cop_param_d_v{v}"
            )
            
            carbon_intensity = st.number_input(
                "Carbon Intensity (gCO₂/kWh)",
                value=float(st.session_state.ates_params.carbon_intensity),
                min_value=0.0,
                step=10.0,
                format="%.2f",
                help="Grid carbon intensity",
                key=f"carbon_intensity_v{v}"
            )
        
        st.session_state['_temp_cop_param_a'] = cop_param_a
        st.session_state['_temp_cop_param_b'] = cop_param_b
        st.session_state['_temp_cop_param_c'] = cop_param_c
        st.session_state['_temp_cop_param_d'] = cop_param_d
        st.session_state['_temp_pump_energy_density'] = pump_energy_density
        st.session_state['_temp_carbon_intensity'] = carbon_intensity

def render_parameter_section_e():
    """E. Auto-calculated Parameters (Display Read-only)"""
    with st.expander("E. Auto-calculated Parameters", expanded=False):
        st.markdown("**These parameters are automatically calculated based on the input parameters above**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input(
                "Water Volumetric Heat Capacity (J/K/m³)",
                value=f"{st.session_state.ates_params.water_volumetric_heat_capacity:,.2f}",
                disabled=True,
                help="Water Density × Water Specific Heat"
            )
            st.text_input(
                "Thermal Recovery Factor Cooling (-)",
                value=f"{st.session_state.ates_params.thermal_recovery_factor_c:.4f}",
                disabled=True,
                help="Calculated from the heating thermal recovery factor and thermal recovery tolerance"
            )
            
            
            
            # st.text_input(
            #     "Number of Cooling Doublets",
            #     value=f"{st.session_state.ates_params.cooling_number_of_doublets}",
            #     disabled=True,
            #     help="Equal to heating doublets"
            # )
        
        with col2:
            st.text_input(
                "Shoulder Days",
                value=f"{st.session_state.ates_params.shoulder_days:.1f}",
                disabled=True,
                help="365 - Heating Days - Cooling Days"
            )
            # st.text_input(
            #     "Cooling Flow Rate per Doublet (m³/hr)",
            #     value=f"{getattr(st.session_state.ates_params, 'cooling_target_avg_flowrate_pd', 0.0):.2f}",
            #     disabled=True,
            #     help="Calculated from energy balance"
            # )
            
            # st.text_input(
            #     "Total Heating Volume (m³)",
            #     value=f"{st.session_state.ates_params.heating_total_produced_volume:,.2f}",
            #     disabled=True,
            #     help="Total produced heating volume"
            # )
            
            # st.text_input(
            #     "Total Cooling Volume (m³)",
            #     value=f"{st.session_state.ates_params.cooling_total_produced_volume:,.2f}",
            #     disabled=True,
            #     help="Total produced cooling volume"
            # )



def render_parameter_section_f():
    """
    F. Thermal Radius Parameters (Feature B, paper eq 34-37)
    """
    v = st.session_state.get('input_widget_version', 0)

    with st.expander("F. Thermal Radius", expanded=False):
        constrain_by_thermal_radius = st.checkbox(
            "Calculate thermal radius",
            value=bool(st.session_state.ates_params.constrain_by_thermal_radius),
            help="Compute and display warm/cool plume thermal radii for this Quick Look case.",
            key=f"constrain_by_thermal_radius_v{v}"
        )

        if constrain_by_thermal_radius:
            col1, col2 = st.columns(2)
            with col1:
                screen_length = st.number_input(
                    "Borehole Screen Length (m)",
                    value=float(st.session_state.ates_params.screen_length),
                    min_value=0.01, step=1.0, format="%.2f",
                    help="Average effective borehole screen length",
                    key=f"screen_length_v{v}"
                )
                aquifer_porosity = st.number_input(
                    "Aquifer Porosity (-)",
                    value=float(st.session_state.ates_params.aquifer_porosity),
                    min_value=0.0, max_value=1.0, step=0.01, format="%.3f",
                    help="Average aquifer porosity",
                    key=f"aquifer_porosity_v{v}"
                )
                max_thermal_radius = st.number_input(
                    "Maximum Thermal Radius (m)",
                    value=float(st.session_state.ates_params.max_thermal_radius),
                    min_value=0.01, step=1.0, format="%.2f",
                    help="Reference threshold for the Quick Look thermal radius comparison",
                    key=f"max_thermal_radius_v{v}"
                )
            with col2:
                rock_specific_heat_capacity = st.number_input(
                    "Rock Specific Heat Capacity (J/kg/°C)",
                    value=float(st.session_state.ates_params.rock_specific_heat_capacity),
                    min_value=0.01, step=10.0, format="%.2f",
                    help="Average aquifer rock specific heat capacity",
                    key=f"rock_specific_heat_capacity_v{v}"
                )
                rock_density = st.number_input(
                    "Rock Density (kg/m³)",
                    value=float(st.session_state.ates_params.rock_density),
                    min_value=0.01, step=10.0, format="%.2f",
                    help="Average aquifer rock density",
                    key=f"rock_density_v{v}"
                )
        else:
            screen_length = float(st.session_state.ates_params.screen_length)
            aquifer_porosity = float(st.session_state.ates_params.aquifer_porosity)
            rock_specific_heat_capacity = float(st.session_state.ates_params.rock_specific_heat_capacity)
            rock_density = float(st.session_state.ates_params.rock_density)
            max_thermal_radius = float(st.session_state.ates_params.max_thermal_radius)

        st.session_state['_temp_constrain_by_thermal_radius'] = constrain_by_thermal_radius
        st.session_state['_temp_screen_length'] = screen_length
        st.session_state['_temp_aquifer_porosity'] = aquifer_porosity
        st.session_state['_temp_rock_specific_heat_capacity'] = rock_specific_heat_capacity
        st.session_state['_temp_rock_density'] = rock_density
        st.session_state['_temp_max_thermal_radius'] = max_thermal_radius


def initialize_temp_variables_from_params():
    """
    Initialize temporary variables from ates_params
    """
    params = st.session_state.ates_params
    
    # A. Basic Physical Parameters
    st.session_state['_temp_aquifer_temp'] = params.aquifer_temp
    st.session_state['_temp_water_density'] = params.water_density
    st.session_state['_temp_water_specific_heat_capacity'] = params.water_specific_heat_capacity
    st.session_state['_temp_thermal_recovery_factor'] = params.thermal_recovery_factor
    st.session_state['_temp_tolerance_in_thermal_recovery'] = params.tolerance_in_thermal_recovery
    st.session_state['_temp_use_volume_balance'] = params.use_volume_balance
    st.session_state['_temp_tolerance_in_volume_balance'] = params.tolerance_in_volume_balance
    
    # B. System Operational Parameters
    st.session_state['_temp_specify_cooling_flowrate'] = params.specify_cooling_flowrate
    st.session_state['_temp_heating_target_avg_flowrate_pd'] = params.heating_target_avg_flowrate_pd
    st.session_state['_temp_cooling_target_avg_flowrate_pd'] = params.cooling_target_avg_flowrate_pd
    st.session_state['_temp_tolerance_in_energy_balance'] = params.tolerance_in_energy_balance
    st.session_state['_temp_heating_number_of_doublets'] = params.heating_number_of_doublets
    st.session_state['_temp_heating_days'] = params.heating_days
    st.session_state['_temp_cooling_days'] = params.cooling_days
    st.session_state['_temp_pump_energy_density'] = params.pump_energy_density
    st.session_state['_temp_heating_ave_injection_temp'] = params.heating_ave_injection_temp
    st.session_state['_temp_heating_temp_to_building'] = params.heating_temp_to_building
    
    # C. COP Parameters
    st.session_state['_temp_cop_param_a'] = params.cop_param_a
    st.session_state['_temp_cop_param_b'] = params.cop_param_b
    st.session_state['_temp_cop_param_c'] = params.cop_param_c
    st.session_state['_temp_cop_param_d'] = params.cop_param_d
    st.session_state['_temp_carbon_intensity'] = params.carbon_intensity
    
    # D. Cooling Side Parameters
    st.session_state['_temp_cooling_ave_injection_temp'] = params.cooling_ave_injection_temp
    st.session_state['_temp_cooling_temp_to_building'] = params.cooling_temp_to_building

    # F. Thermal radius parameters (Feature B)
    st.session_state['_temp_constrain_by_thermal_radius'] = params.constrain_by_thermal_radius
    st.session_state['_temp_screen_length'] = params.screen_length
    st.session_state['_temp_aquifer_porosity'] = params.aquifer_porosity
    st.session_state['_temp_rock_specific_heat_capacity'] = params.rock_specific_heat_capacity
    st.session_state['_temp_rock_density'] = params.rock_density
    st.session_state['_temp_max_thermal_radius'] = params.max_thermal_radius



# VALIDATION AND CALCULATION

def validate_parameters():
    """Validate parameters"""
    errors = []
    params = st.session_state.ates_params
    
    if params.heating_ave_injection_temp >= params.aquifer_temp:
        errors.append("Cool well injection temperature must be less than aquifer temperature")
    
    if params.cooling_ave_injection_temp <= params.aquifer_temp:
        errors.append("Warm well injection temperature must be greater than aquifer temperature")

    
    if params.heating_temp_to_building < params.aquifer_temp:
        errors.append(f"Building heating temperature ({params.heating_temp_to_building:.1f}°C) must be >= aquifer temperature ({params.aquifer_temp:.1f}°C)")
    
    if params.cooling_temp_to_building > params.aquifer_temp:
        errors.append(f"Building cooling temperature ({params.cooling_temp_to_building:.1f}°C) must be <= aquifer temperature ({params.aquifer_temp:.1f}°C)")
    
    total_days = params.heating_days + params.cooling_days
    if total_days > 365:
        errors.append(f"The sum of heating and cooling days cannot exceed 365 (Current: {total_days:.1f})")
    
    if params.thermal_recovery_factor < 0 or params.thermal_recovery_factor > 1:
        errors.append("Thermal recovery factor must be between 0 and 1")
    
    if params.cop_param_b <= 0:
        errors.append("COP parameter B must be a positive number")
    
    return errors

def perform_calculation():
    """Execute ATES calculation with proper distribution initialization"""
    try:
        # Update all parameters from temporary variables before calculation
        update_all_parameters_from_temp()
        start_time = time.time()
        
        # Validate parameters
        errors = validate_parameters()
        if errors:
            for error in errors:
                st.error(f"Error: {error}")
            return False
        
        # Create calculator and perform calculation
        calculator = ATESCalculator(st.session_state.ates_params)
        results = calculator.calculate()
        
        # Check Eq31 physical constraint
        p = st.session_state.ates_params
        violations = []
        if not (p.heating_ave_injection_temp <= results.cooling_physical_production_temp):
            violations.append(f"Cool well production temp ({results.cooling_physical_production_temp:.2f}°C) < cool well injection temp ({p.heating_ave_injection_temp:.2f}°C)")
        if not (results.cooling_physical_production_temp <= p.aquifer_temp):
            violations.append(f"Cool well production temp ({results.cooling_physical_production_temp:.2f}°C) > aquifer temp ({p.aquifer_temp:.2f}°C)")
        if not (p.aquifer_temp <= results.heating_physical_production_temp):
            violations.append(f"Warm well production temp ({results.heating_physical_production_temp:.2f}°C) < aquifer temp ({p.aquifer_temp:.2f}°C)")
        if not (results.heating_physical_production_temp <= p.cooling_ave_injection_temp):
            violations.append(f"Warm well production temp ({results.heating_physical_production_temp:.2f}°C) > warm well injection temp ({p.cooling_ave_injection_temp:.2f}°C)")
        
        if violations:
            message = "Calculation stopped: physically non-viable parameters. " + "; ".join(violations)
            st.error(message)
            return False 
        
        # Save results
        st.session_state.results = results
        st.session_state.calculation_count += 1
        
        # Initialize or update probability distributions after successful calculation
        calc_time = time.time() - start_time
        st.success(f"Calculation complete! Time taken: {calc_time:.3f} seconds")
        return True
        
    except Exception as e:
        st.error(f"Calculation failed: {str(e)}")
        return False

# results display

def render_heating_results(results):
    """
    Render heating results
    """
    with st.expander("Heating Results", expanded=True):
        # key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "System COP",
                f"{results.heating_system_cop:.2f}",
                help="Heating system Coefficient of Performance"
            )
        
        with col2:
            st.metric(
                "Building Energy",
                f"{results.heating_annual_energy_building_GWhth:.2f} GWh",
                help="Annual energy supplied to the building"
            )
        
        with col3:
            st.metric(
                "Electrical Energy",
                f"{results.heating_annual_elec_energy_GWhe:.2f} GWh",
                help="Annual electrical energy consumption"
            )
        
        with col4:
            st.metric(
                "CO₂ Emissions",
                f"{results.heating_co2_emissions_per_thermal:.0f} gCO₂/kWh",
                help="Carbon emissions"
            )
        
        # Detailed results table
        heating_data = []
        heating_params = [
            ("Total Energy Stored (J)", results.heating_total_energy_stored, "J"),
            ("Stored Energy Recovered (J)", results.heating_stored_energy_recovered, "J"),
            ("Total Flow Rate (m³/hr)", results.heating_total_flow_rate_m3hr, "m³/hr"),
            ("Total Flow Rate (l/s)", results.heating_total_flow_rate_ls, "l/s"),
            ("Total Flow Rate (m³/s)", results.heating_total_flow_rate_m3s, "m³/s"),
            ("Average Production Temperature", results.heating_physical_production_temp, "°C"),
            ("Average Temperature Change Across Heat Exchanger", results.heating_ave_temp_change_across_HX, "°C"),
            ("Temperature Change Induced by HP", results.heating_temp_change_induced_HP, "°C"),
            ("Heat Pump COP", results.heating_heat_pump_COP, "-"),
            ("Heat Pump Factor (ehp)", results.heating_ehp, "-"),
            ("Average Power to Heat Exchanger (W)", results.heating_ave_power_to_HX_W, "W"),
            ("Average Power to Heat Exchanger (MW)", results.heating_ave_power_to_HX_MW, "MW"),
            ("Annual Energy from Aquifer (J)", results.heating_annual_energy_aquifer_J, "J"),
            ("Annual Energy from Aquifer (kWhth)", results.heating_annual_energy_aquifer_kWhth, "kWhth"),
            ("Annual Energy from Aquifer (GWhth)", results.heating_annual_energy_aquifer_GWhth, "GWhth"),
            ("Monthly Energy to Heat Exchanger", results.heating_monthly_to_HX, "GWh/month"),
            ("Average Power to Building (W)", results.heating_ave_power_to_building_W, "W"),
            ("Average Power to Building (MW)", results.heating_ave_power_to_building_MW, "MW"),
            ("Annual Energy to Building (J)", results.heating_annual_energy_building_J, "J"),
            ("Annual Energy to Building (kWhth)", results.heating_annual_energy_building_kWhth, "kWhth"),
            ("Annual Energy to Building (GWhth)", results.heating_annual_energy_building_GWhth, "GWhth"),
            ("Monthly Energy to Building", results.heating_monthly_to_building, "GWh/month"),
            ("Electrical Energy to Hydraulic Pumps", results.heating_elec_energy_hydraulic_pumps, "J"),
            ("Electrical Energy to Heat Pump", results.heating_elec_energy_HP, "J"),
            ("Annual Electrical Energy (J)", results.heating_annual_elec_energy_J, "J"),
            ("Annual Electrical Energy (MWhe)", results.heating_annual_elec_energy_MWhe, "MWhe"),
            ("Annual Electrical Energy (GWhe)", results.heating_annual_elec_energy_GWhe, "GWhe"),
            ("System COP", results.heating_system_cop, "-"),
            ("Electrical Energy per Thermal", results.heating_elec_energy_per_thermal, "kWhe/kWhth"),
            ("CO₂ Emissions per Thermal", results.heating_co2_emissions_per_thermal, "gCO₂/kWhth"),
        ]

        for name, value, unit in heating_params:
            if isinstance(value, float):
                if abs(value) > 1e6:
                    formatted_value = f"{value:.2e}"
                elif abs(value) > 100:
                    formatted_value = f"{value:.0f}"
                else:
                    formatted_value = f"{value:.3f}"
            else:
                formatted_value = str(value)

            heating_data.append({
                "Parameter": name,
                "Value": formatted_value,
                "Unit": unit
            })

        df_heating = pd.DataFrame(heating_data)
        st.dataframe(df_heating, width="stretch", hide_index=True)

def render_cooling_results(results):
    """
    Render cooling results
    """
    with st.expander("Cooling Results", expanded=True):
        is_direct_cooling = getattr(results, 'cooling_direct_mode', False)
        
        # one more check with direct mode and the heat pump COP
        if not hasattr(results, 'cooling_direct_mode'):
            is_direct_cooling = (results.cooling_heat_pump_COP == float('inf'))
        
        if is_direct_cooling:
            st.success("Direct Cooling Mode Active - Production temperature sufficient for direct cooling")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "System COP",
                f"{results.cooling_system_cop:.2f}", 
                help="Cooling system Coefficient of Performance"
            )
        
        with col2:
            st.metric(
                "Building Energy",
                f"{results.cooling_annual_energy_building_GWhth:.2f} GWh",
                help="Annual energy supplied to the building"
            )
        
        with col3:
            st.metric(
                "Electrical Energy",
                f"{results.cooling_annual_elec_energy_GWhe:.2f} GWh",
                help="Annual electrical energy consumption"
            )
        
        with col4:
            st.metric(
                "CO₂ Emissions",
                f"{results.cooling_co2_emissions_per_thermal:.0f} gCO₂/kWh",
                help="Carbon emissions"
            )
        
        # cooling data result
        cooling_data = []
        cooling_params = [
            ("Total Energy Stored (J)", results.cooling_total_energy_stored, "J"),
            ("Stored Energy Recovered (J)", results.cooling_stored_energy_recovered, "J"),
            ("Target Flow Rate per Borehole (m³/hr)", st.session_state.ates_params.cooling_target_avg_flowrate_pd, "m³/hr"),
            ("Total Flow Rate (m³/hr)", results.cooling_total_flow_rate_m3hr, "m³/hr"),
            ("Total Flow Rate (l/s)", results.cooling_total_flow_rate_ls, "l/s"),
            ("Total Flow Rate (m³/s)", results.cooling_total_flow_rate_m3s, "m³/s"),
            ("Average Production Temperature", results.cooling_physical_production_temp, "°C"),
            ("Average Temperature Change Across Heat Exchanger", results.cooling_ave_temp_change_across_HX, "°C"),
            ("Temperature Change Induced by HP", results.cooling_temp_change_induced_HP, "°C"),
            ("Heat Pump COP", results.cooling_heat_pump_COP, "-"),
            ("Heat Pump Factor (ehp)", results.cooling_ehp, "-"),
            ("Average Power to Heat Exchanger (W)", results.cooling_ave_power_to_HX_W, "W"),
            ("Average Power to HHeat Exchanger (MW)", results.cooling_ave_power_to_HX_MW, "MW"),
            ("Annual Energy from Aquifer (J)", results.cooling_annual_energy_aquifer_J, "J"),
            ("Annual Energy from Aquifer (kWhth)", results.cooling_annual_energy_aquifer_kWhth, "kWhth"),
            ("Annual Energy from Aquifer (GWhth)", results.cooling_annual_energy_aquifer_GWhth, "GWhth"),
            ("Monthly Energy to Heat Exchanger", results.cooling_monthly_to_HX, "GWh/month"),
            ("Average Power to Building (W)", results.cooling_ave_power_to_building_W, "W"),
            ("Average Power to Building (MW)", results.cooling_ave_power_to_building_MW, "MW"),
            ("Annual Energy to Building (J)", results.cooling_annual_energy_building_J, "J"),
            ("Annual Energy to Building (kWhth)", results.cooling_annual_energy_building_kWhth, "kWhth"),
            ("Annual Energy to Building (GWhth)", results.cooling_annual_energy_building_GWhth, "GWhth"),
            ("Monthly Energy to Building", results.cooling_monthly_to_building, "GWh/month"),
            ("Electrical Energy to Hydraulic Pumps", results.cooling_elec_energy_hydraulic_pumps, "J"),
            ("Electrical Energy to Heat Pump", results.cooling_elec_energy_HP, "J"),
            ("Annual Electrical Energy (J)", results.cooling_annual_elec_energy_J, "J"),
            ("Annual Electrical Energy (MWhe)", results.cooling_annual_elec_energy_MWhe, "MWhe"),
            ("Annual Electrical Energy (GWhe)", results.cooling_annual_elec_energy_GWhe, "GWhe"),
            ("System COP", results.cooling_system_cop, "-"),
            ("Electrical Energy per Thermal", results.cooling_elec_energy_per_thermal, "kWhe/kWhth"),
            ("CO₂ Emissions per Thermal", results.cooling_co2_emissions_per_thermal, "gCO₂/kWhth"),
        ]

        for name, value, unit in cooling_params:
            if isinstance(value, float):
                if value == float('inf') and name in ["Cooling Heat Pump COP", "Cooling Heat Pump Factor (ehp)"]:
                    formatted_value = "Direct Mode"
                elif abs(value) > 1e6:
                    formatted_value = f"{value:.2e}"
                elif abs(value) > 100:
                    formatted_value = f"{value:.0f}"
                else:
                    formatted_value = f"{value:.3f}"
            else:
                formatted_value = str(value)
            
            cooling_data.append({
                "Parameter": name,
                "Value": formatted_value,
                "Unit": unit
            })

        df_cooling = pd.DataFrame(cooling_data)
        st.dataframe(df_cooling, width="stretch", hide_index=True)

def render_system_balance_and_volumes(results, params):
    """
    Render system balance and groundwater volumes metrics
    """
    with st.expander("System Balance & Groundwater Volumes", expanded=True):
        
        # Balance ratios
        # st.markdown("**System Balance**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Energy Balance Ratio (EBR)",
                f"{results.energy_balance_ratio:.3f}",
                help="Ratio of cooling energy to heating energy stored"
            )
        
        with col2:
            st.metric(
                "Volume Balance Ratio (VBR)",
                f"{results.volume_balance_ratio:.3f}",
                help="Ratio of cooling volume to heating volume"
            )
        
        # Groundwater volumes
        # st.markdown("**Groundwater Volumes**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Heating Produced Volume",
                f"{params.heating_total_produced_volume:,.0f} m³",
                help="Total groundwater volume produced during heating season"
            )
        
        with col2:
            st.metric(
                "Cooling Produced Volume",
                f"{params.cooling_total_produced_volume:,.0f} m³",
                help="Total groundwater volume produced during cooling season"
            )
        
        # # Detailed table
        # sustainability_data = []
        # sustainability_params = [
        #     ("Energy Balance Ratio (EBR)", results.energy_balance_ratio, "-"),
        #     ("Volume Balance Ratio (VBR)", results.volume_balance_ratio, "-"),
        # ]
        
        # for name, value, unit in sustainability_params:
        #     if isinstance(value, float):
        #         formatted_value = f"{value:.3f}"
        #     else:
        #         formatted_value = str(value)
            
        #     sustainability_data.append({
        #         "Parameter": name,
        #         "Value": formatted_value,
        #         "Unit": unit
        #     })
        
        # df_sustainability = pd.DataFrame(sustainability_data)
        # st.dataframe(df_sustainability, width="stretch", hide_index=True)


def render_thermal_radius_results(results, params):
    """Render thermal radius results (Feature B)."""
    with st.expander("Thermal Radius", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Warm Plume Radius",
                      f"{results.thermal_radius_h:.2f} m",
                      help="Thermal radius of the warm plume")
        with col2:
            st.metric("Cool Plume Radius",
                      f"{results.thermal_radius_c:.2f} m",
                      help="Thermal radius of the cool plume")
        with col3:
            st.metric("Max Allowed Thermal Radius",
                      f"{params.max_thermal_radius:.2f} m",
                      help="Reference threshold for this Quick Look comparison")

        exceeded = (results.thermal_radius_h > params.max_thermal_radius or
                    results.thermal_radius_c > params.max_thermal_radius)
        if exceeded:
            st.warning("Thermal radius exceeds the maximum.")
        else:
            st.success("Thermal radius is within the maximum.")


# MAIN APPLICATION

def main():
    """
    Main function with case management integration 
    """
    # initialize session state and distributions
    initialize_session_state()
    if 'param_distributions' not in st.session_state:
        initialize_default_distributions()
    
    # Check and initialize temporary variables
    temp_keys_exist = any(str(key).startswith('_temp_') for key in st.session_state.keys() if isinstance(key, str))
    if not temp_keys_exist:
        initialize_temp_variables_from_params()
    
    if 'sync_enabled' not in st.session_state:
        st.session_state.sync_enabled = False
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Quick Look'
    
    # get app state for case management
    app_state = get_app_state()
    
    # main page title
    st.title("'Quick Look' Deterministic Analysis")
    st.markdown("**Imperial Aquifer Thermal Energy Storage System Calculation Tool**")
    
    # create two-column layout
    col_params, col_results = st.columns([1.2, 1])
    
    with col_params:
        st.header("Input Parameters")
        
        # render parameter sections
        render_parameter_section_a()
        render_parameter_section_b()
        render_parameter_section_c()
        render_parameter_section_d()
        render_parameter_section_e()
        render_parameter_section_f()
        
        # Operation buttons
        st.markdown("### Operations")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Calculate", type="primary", width="stretch"):
                if perform_calculation():
                    st.rerun()
        
        with col2:
            if st.button("Reset", width="stretch"):
                # Try to restore case snapshot first (for loaded cases)
                app_state = get_app_state()
                if app_state.restore_case_snapshot():
                    # Reset to loaded case state
                    st.session_state['case_modified'] = False
                    st.success("Parameters reset to loaded case state")
                else:
                    # No snapshot, reset to default
                    from tool.core.ates_calculator import ATESParameters
                    st.session_state.ates_params = ATESParameters()
                    
                    # Clear calculation results
                    st.session_state['results'] = None
                    st.session_state['calculation_count'] = 0
                    st.session_state['calculation_status'] = 'not_started'
                    
                    # Reinitialize distributions
                    initialize_default_distributions()
                    
                    # Sync temp variables
                    initialize_temp_variables_from_params()
                    
                    # Increment widget version to force refresh
                    st.session_state['input_widget_version'] = st.session_state.get('input_widget_version', 0) + 1
                    
                    st.session_state['case_modified'] = False
                    st.success("Parameters reset to default values")
                
                st.rerun()
        
        with col3:
            if st.button("Validate", width="stretch"):
                # Update parameters first, then validate
                update_all_parameters_from_temp()
                errors = validate_parameters()
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    # Eq31 physical viability check
                    p = st.session_state.ates_params
                    results = ATESCalculator(p).calculate()
                    violations = []
                    if not (p.heating_ave_injection_temp <= results.cooling_physical_production_temp):
                        violations.append(f"Cool well production temp ({results.cooling_physical_production_temp:.2f}°C) < cool well injection temp ({p.heating_ave_injection_temp:.2f}°C)")
                    if not (results.cooling_physical_production_temp <= p.aquifer_temp):
                        violations.append(f"Cool well production temp ({results.cooling_physical_production_temp:.2f}°C) > aquifer temp ({p.aquifer_temp:.2f}°C)")
                    if not (p.aquifer_temp <= results.heating_physical_production_temp):
                        violations.append(f"Warm well production temp ({results.heating_physical_production_temp:.2f}°C) < aquifer temp ({p.aquifer_temp:.2f}°C)")
                    if not (results.heating_physical_production_temp <= p.cooling_ave_injection_temp):
                        violations.append(f"Warm well production temp ({results.heating_physical_production_temp:.2f}°C) > warm well injection temp ({p.cooling_ave_injection_temp:.2f}°C)")
                    
                    if violations:
                        st.error("Parameters physically non-viable. " + "; ".join(violations))
                    else:
                        st.success("All parameters are valid")
    
    with col_results:
        st.header("Calculation Results")
        
        if st.session_state.results is None:
            st.info("Configure parameters on the left and click 'Calculate' to view results")
        else:
            results = st.session_state.results
            
            # Render results
            render_heating_results(results)
            render_cooling_results(results)
            render_system_balance_and_volumes(results, st.session_state.ates_params)
            if st.session_state.ates_params.constrain_by_thermal_radius:
                render_thermal_radius_results(results, st.session_state.ates_params)

if __name__ == "__main__":
    main()
