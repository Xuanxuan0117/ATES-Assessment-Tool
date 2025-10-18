"""
Screen 1 - Quick Calculation 
"""
import streamlit as st
import pandas as pd
import time
import json

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
        
        # B. System Operational Parameters
        ('heating_target_avg_flowrate_pd', '_temp_heating_target_avg_flowrate_pd'),
        ('tolerance_in_energy_balance', '_temp_tolerance_in_energy_balance'),
        ('heating_number_of_doublets', '_temp_heating_number_of_doublets'),
        ('heating_months', '_temp_heating_months'),
        ('cooling_months', '_temp_cooling_months'),
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
            else:
                if abs(old_value - new_value) > 1e-9:
                    setattr(st.session_state.ates_params, param_name, new_value)
                    has_changes = True
    
    # If changes detected, recalculate derived parameters and mark case as modified
    if has_changes:
        st.session_state.ates_params.__post_init__()
        mark_case_modified()
        
        # Synchronize to probability distribution settings if sync is enabled
        if st.session_state.get('sync_enabled', True):
            sync_all_params_to_distributions()

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
            
            # update range if it was auto-calculated
            if abs(dist['min'] - dist['max']) < 1e-6:
                dist['min'] = value * 0.8
                dist['max'] = value * 1.2
            
            dist['std'] = max(value * 0.1, 0.01)

def sync_all_params_to_distributions():
    """
    Synchronize all parameters to probability distribution settings
    """
    if 'param_distributions' not in st.session_state:
        return
    
    param_names = [
        'aquifer_temp', 'water_density', 'water_specific_heat_capacity', 'thermal_recovery_factor',
        'heating_target_avg_flowrate_pd', 'tolerance_in_energy_balance', 'heating_number_of_doublets',
        'heating_months', 'cooling_months', 'pump_energy_density',
        'heating_ave_injection_temp', 'heating_temp_to_building',
        'cop_param_a', 'cop_param_b', 'cop_param_c', 'cop_param_d', 'carbon_intensity',
        'cooling_ave_injection_temp', 'cooling_temp_to_building'
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
            'heating_months', 'cooling_months', 'pump_energy_density', 
            'heating_ave_injection_temp', 'heating_temp_to_building',
            'cop_param_a', 'cop_param_b', 'cop_param_c', 'cop_param_d',
            'carbon_intensity', 'cooling_ave_injection_temp', 'cooling_temp_to_building'
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
    else:
        # Update existing distributions with current parameter values
        sync_all_params_to_distributions()

# Parameter input sections

def render_parameter_section_a():
    """
    Physical Parameters (3 parameters)
    """
    with st.expander("Physical Parameters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            aquifer_temp = st.number_input(
                "Aquifer Temperature (°C)",
                value=float(st.session_state.ates_params.aquifer_temp),
                min_value=5.0,
                max_value=25.0,
                step=0.1,
                format="%.2f",
                help="Aquifer temperature"
            )
            
            water_density = st.number_input(
                "Water Density (kg/m³)",
                value=float(st.session_state.ates_params.water_density),
                min_value=995.0,
                max_value=1005.0,
                step=0.1,
                format="%.2f",
                help="Length of the cooling season"
            )
        
        with col2:
            water_specific_heat_capacity = st.number_input(
                "Water Specific Heat Capacity (J/kg/K)",
                value=float(st.session_state.ates_params.water_specific_heat_capacity),
                min_value=4000.0,
                max_value=4300.0,
                step=1.0,
                format="%.2f",
                help="Water specific heat capacity"
            )
        
        st.session_state['_temp_aquifer_temp'] = aquifer_temp
        st.session_state['_temp_water_density'] = water_density
        st.session_state['_temp_water_specific_heat_capacity'] = water_specific_heat_capacity


def render_parameter_section_b():
    """
    Demand Parameters (4 parameters) 
    """
    with st.expander("Demand Parameters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            heating_months = st.number_input(
                "Heating Months",
                value=float(st.session_state.ates_params.heating_months),
                min_value=1.0,
                max_value=10.0,
                step=0.1,
                format="%.2f",
                help="Length of the heating season"
            )
            
            heating_temp_to_building = st.number_input(
                "Building Heating Temperature (°C)",
                value=float(st.session_state.ates_params.heating_temp_to_building),
                min_value=40.0,
                max_value=80.0,
                step=1.0,
                format="%.2f",
                help="Building heating temperature"
            )
        
        with col2:
            cooling_months = st.number_input(
                "Cooling Months",
                value=float(st.session_state.ates_params.cooling_months),
                min_value=0.5,
                max_value=8.0,
                step=0.1,
                format="%.2f",
                help="Length of the cooling season"
            )
            
            cooling_temp_to_building = st.number_input(
                "Building Cooling Temperature (°C)",
                value=float(st.session_state.ates_params.cooling_temp_to_building),
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                format="%.2f",
                help="Building cooling temperature"
            )
        
        st.session_state['_temp_heating_months'] = heating_months
        st.session_state['_temp_cooling_months'] = cooling_months
        st.session_state['_temp_heating_temp_to_building'] = heating_temp_to_building
        st.session_state['_temp_cooling_temp_to_building'] = cooling_temp_to_building


def render_parameter_section_c():
    """
    ATES System Operation (6 parameters) 
    """
    with st.expander("ATES System Operation", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            heating_target_avg_flowrate_pd = st.number_input(
                "Target Flow Rate Heating (m³/hr)",
                value=float(st.session_state.ates_params.heating_target_avg_flowrate_pd),
                min_value=10.0,
                max_value=200.0,
                step=1.0,
                format="%.2f",
                help="Target flow rate for heating per doublet"
            )
            
            heating_number_of_doublets = st.number_input(
                "Number of Doublets",
                value=int(st.session_state.ates_params.heating_number_of_doublets),
                min_value=1,
                max_value=100,
                step=1,
                help="Number of well doublets"
            )
            
            heating_ave_injection_temp = st.number_input(
                "Heating Injection Temperature (°C)",
                value=float(st.session_state.ates_params.heating_ave_injection_temp),
                min_value=5.0,
                max_value=15.0,
                step=0.1,
                format="%.2f",
                help="Heating injection temperature (< Aquifer Temperature)"
            )
        
        with col2:
            thermal_recovery_factor = st.number_input(
                "Thermal Recovery Factor (-)",
                value=float(st.session_state.ates_params.thermal_recovery_factor),
                min_value=0.1,
                max_value=1.0,
                step=0.01,
                format="%.2f",
                help="Thermal recovery efficiency"
            )
            
            tolerance_in_energy_balance = st.number_input(
                "Energy Balance Tolerance (-)",
                value=float(st.session_state.ates_params.tolerance_in_energy_balance),
                min_value=0.05,
                max_value=0.5,
                step=0.01,
                format="%.2f",
                help="Energy balance tolerance"
            )
            
            cooling_ave_injection_temp = st.number_input(
                "Cooling Injection Temperature (°C)",
                value=float(st.session_state.ates_params.cooling_ave_injection_temp),
                min_value=15.0,
                max_value=35.0,
                step=0.1,
                format="%.2f",
                help="Cooling injection temperature (> Aquifer Temperature)"
            )
        
        st.session_state['_temp_heating_target_avg_flowrate_pd'] = heating_target_avg_flowrate_pd
        st.session_state['_temp_heating_number_of_doublets'] = heating_number_of_doublets
        st.session_state['_temp_thermal_recovery_factor'] = thermal_recovery_factor
        st.session_state['_temp_tolerance_in_energy_balance'] = tolerance_in_energy_balance
        st.session_state['_temp_heating_ave_injection_temp'] = heating_ave_injection_temp
        st.session_state['_temp_cooling_ave_injection_temp'] = cooling_ave_injection_temp


def render_parameter_section_d():
    """
    Heat Pump and Carbon Intensity (6 parameters) 
    """
    with st.expander("Heat Pump and Carbon Intensity", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            cop_param_a = st.number_input(
                "COP Parameter A (-)",
                value=float(st.session_state.ates_params.cop_param_a),
                min_value=50.0,
                max_value=200.0,
                step=1.0,
                format="%.2f",
                help="COP model parameter A"
            )
            
            cop_param_c = st.number_input(
                "COP Parameter C (-)",
                value=float(st.session_state.ates_params.cop_param_c),
                min_value=-0.2,
                max_value=0.2,
                step=0.01,
                format="%.2f",
                help="COP model parameter C"
            )
            
            pump_energy_density = st.number_input(
                "Hydraulic Pump Energy Density (kJ/m³)",
                value=float(st.session_state.ates_params.pump_energy_density),
                min_value=200.0,
                max_value=1500.0,
                step=10.0,
                format="%.2f",
                help="Hydraulic pump energy density"
            )
        
        with col2:
            cop_param_b = st.number_input(
                "COP Parameter B (-)",
                value=float(st.session_state.ates_params.cop_param_b),
                min_value=0.5,
                max_value=3.0,
                step=0.1,
                format="%.2f",
                help="COP model parameter B (must be positive)"
            )
            
            cop_param_d = st.number_input(
                "COP Parameter D (-)",
                value=float(st.session_state.ates_params.cop_param_d),
                min_value=1.0,
                max_value=15.0,
                step=0.1,
                format="%.2f",
                help="COP model parameter D"
            )
            
            carbon_intensity = st.number_input(
                "Carbon Intensity (gCO₂/kWh)",
                value=float(st.session_state.ates_params.carbon_intensity),
                min_value=0.0,
                max_value=1000.0,
                step=10.0,
                format="%.2f",
                help="Grid carbon intensity"
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
                "Shoulder Months",
                value=f"{st.session_state.ates_params.shoulder_months:.1f}",
                disabled=True,
                help="12 - Heating Months - Cooling Months"
            )
            
            st.text_input(
                "Number of Cooling Doublets",
                value=f"{st.session_state.ates_params.cooling_number_of_doublets}",
                disabled=True,
                help="Equal to heating doublets"
            )
        
        with col2:
            st.text_input(
                "Cooling Flow Rate per Doublet (m³/hr)",
                value=f"{getattr(st.session_state.ates_params, 'cooling_target_avg_flowrate_pd', 0.0):.2f}",
                disabled=True,
                help="Calculated from energy balance"
            )
            
            st.text_input(
                "Total Heating Volume (m³)",
                value=f"{st.session_state.ates_params.heating_total_produced_volume:,.2f}",
                disabled=True,
                help="Total produced heating volume"
            )
            
            st.text_input(
                "Total Cooling Volume (m³)",
                value=f"{st.session_state.ates_params.cooling_total_produced_volume:,.2f}",
                disabled=True,
                help="Total produced cooling volume"
            )



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
    
    # B. System Operational Parameters
    st.session_state['_temp_heating_target_avg_flowrate_pd'] = params.heating_target_avg_flowrate_pd
    st.session_state['_temp_tolerance_in_energy_balance'] = params.tolerance_in_energy_balance
    st.session_state['_temp_heating_number_of_doublets'] = params.heating_number_of_doublets
    st.session_state['_temp_heating_months'] = params.heating_months
    st.session_state['_temp_cooling_months'] = params.cooling_months
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



# VALIDATION AND CALCULATION

def validate_parameters():
    """Validate parameters"""
    errors = []
    params = st.session_state.ates_params
    
    if params.heating_ave_injection_temp >= params.aquifer_temp:
        errors.append("Heating injection temperature must be less than aquifer temperature")
    
    if params.cooling_ave_injection_temp <= params.aquifer_temp:
        errors.append("Cooling injection temperature must be greater than aquifer temperature")
    
    total_months = params.heating_months + params.cooling_months
    if total_months > 12:
        errors.append(f"The sum of heating and cooling months cannot exceed 12 (Current: {total_months:.1f})")
    
    if params.thermal_recovery_factor <= 0 or params.thermal_recovery_factor > 1:
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
        
        # Save results
        st.session_state.results = results
        st.session_state.calculation_count += 1
        
        # Initialize or update probability distributions after successful calculation
        initialize_default_distributions()
        
        # Mark case as modified for new calculation result
        mark_case_modified()
        
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
            ("Heating Total Energy Stored (J)", results.heating_total_energy_stored, "J"),
            ("Heating Stored Energy Recovered (J)", results.heating_stored_energy_recovered, "J"),
            ("Heating Total Flow Rate (m³/hr)", results.heating_total_flow_rate_m3hr, "m³/hr"),
            ("Heating Total Flow Rate (l/s)", results.heating_total_flow_rate_ls, "l/s"),
            ("Heating Total Flow Rate (m³/s)", results.heating_total_flow_rate_m3s, "m³/s"),
            ("Heating Average Production Temperature", results.heating_ave_production_temp, "°C"),
            ("Heating Average Temperature Change Across HX", results.heating_ave_temp_change_across_HX, "°C"),
            ("Heating Temperature Change Induced by HP", results.heating_temp_change_induced_HP, "°C"),
            ("Heating Heat Pump COP", results.heating_heat_pump_COP, "-"),
            ("Heating Heat Pump Factor (ehp)", results.heating_ehp, "-"),
            ("Heating Average Power to HX (W)", results.heating_ave_power_to_HX_W, "W"),
            ("Heating Average Power to HX (MW)", results.heating_ave_power_to_HX_MW, "MW"),
            ("Heating Annual Energy from Aquifer (J)", results.heating_annual_energy_aquifer_J, "J"),
            ("Heating Annual Energy from Aquifer (kWhth)", results.heating_annual_energy_aquifer_kWhth, "kWhth"),
            ("Heating Annual Energy from Aquifer (GWhth)", results.heating_annual_energy_aquifer_GWhth, "GWhth"),
            ("Heating Monthly Energy to HX", results.heating_monthly_to_HX, "GWh/month"),
            ("Energy Balance Ratio (EBR)", results.energy_balance_ratio, "-"),
            ("Volume Balance Ratio (VBR)", results.volume_balance_ratio, "-"),
            ("Heating Average Power to Building (W)", results.heating_ave_power_to_building_W, "W"),
            ("Heating Average Power to Building (MW)", results.heating_ave_power_to_building_MW, "MW"),
            ("Heating Annual Energy to Building (J)", results.heating_annual_energy_building_J, "J"),
            ("Heating Annual Energy to Building (kWhth)", results.heating_annual_energy_building_kWhth, "kWhth"),
            ("Heating Annual Energy to Building (GWhth)", results.heating_annual_energy_building_GWhth, "GWhth"),
            ("Heating Monthly Energy to Building", results.heating_monthly_to_building, "GWh/month"),
            ("Heating Electrical Energy to Hydraulic Pumps", results.heating_elec_energy_hydraulic_pumps, "J"),
            ("Heating Electrical Energy to Heat Pump", results.heating_elec_energy_HP, "J"),
            ("Heating Annual Electrical Energy (J)", results.heating_annual_elec_energy_J, "J"),
            ("Heating Annual Electrical Energy (MWhe)", results.heating_annual_elec_energy_MWhe, "MWhe"),
            ("Heating Annual Electrical Energy (GWhe)", results.heating_annual_elec_energy_GWhe, "GWhe"),
            ("Heating System COP", results.heating_system_cop, "-"),
            ("Heating Electrical Energy per Thermal", results.heating_elec_energy_per_thermal, "kWhe/kWhth"),
            ("Heating CO₂ Emissions per Thermal", results.heating_co2_emissions_per_thermal, "gCO₂/kWhth"),
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
        st.dataframe(df_heating, use_container_width=True, hide_index=True)

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
            ("Cooling Total Energy Stored (J)", results.cooling_total_energy_stored, "J"),
            ("Cooling Stored Energy Recovered (J)", results.cooling_stored_energy_recovered, "J"),
            ("Cooling Total Flow Rate (m³/hr)", results.cooling_total_flow_rate_m3hr, "m³/hr"),
            ("Cooling Total Flow Rate (l/s)", results.cooling_total_flow_rate_ls, "l/s"),
            ("Cooling Total Flow Rate (m³/s)", results.cooling_total_flow_rate_m3s, "m³/s"),
            ("Cooling Average Production Temperature", results.cooling_ave_production_temp, "°C"),
            ("Cooling Average Temperature Change Across HX", results.cooling_ave_temp_change_across_HX, "°C"),
            ("Cooling Temperature Change Induced by HP", results.cooling_temp_change_induced_HP, "°C"),
            ("Cooling Heat Pump COP", results.cooling_heat_pump_COP, "-"),
            ("Cooling Heat Pump Factor (ehp)", results.cooling_ehp, "-"),
            ("Cooling Average Power to HX (W)", results.cooling_ave_power_to_HX_W, "W"),
            ("Cooling Average Power to HX (MW)", results.cooling_ave_power_to_HX_MW, "MW"),
            ("Cooling Annual Energy from Aquifer (J)", results.cooling_annual_energy_aquifer_J, "J"),
            ("Cooling Annual Energy from Aquifer (kWhth)", results.cooling_annual_energy_aquifer_kWhth, "kWhth"),
            ("Cooling Annual Energy from Aquifer (GWhth)", results.cooling_annual_energy_aquifer_GWhth, "GWhth"),
            ("Cooling Monthly Energy to HX", results.cooling_monthly_to_HX, "GWh/month"),
            ("Cooling Average Power to Building (W)", results.cooling_ave_power_to_building_W, "W"),
            ("Cooling Average Power to Building (MW)", results.cooling_ave_power_to_building_MW, "MW"),
            ("Cooling Annual Energy to Building (J)", results.cooling_annual_energy_building_J, "J"),
            ("Cooling Annual Energy to Building (kWhth)", results.cooling_annual_energy_building_kWhth, "kWhth"),
            ("Cooling Annual Energy to Building (GWhth)", results.cooling_annual_energy_building_GWhth, "GWhth"),
            ("Cooling Monthly Energy to Building", results.cooling_monthly_to_building, "GWh/month"),
            ("Cooling Electrical Energy to Hydraulic Pumps", results.cooling_elec_energy_hydraulic_pumps, "J"),
            ("Cooling Electrical Energy to Heat Pump", results.cooling_elec_energy_HP, "J"),
            ("Cooling Annual Electrical Energy (J)", results.cooling_annual_elec_energy_J, "J"),
            ("Cooling Annual Electrical Energy (MWhe)", results.cooling_annual_elec_energy_MWhe, "MWhe"),
            ("Cooling Annual Electrical Energy (GWhe)", results.cooling_annual_elec_energy_GWhe, "GWhe"),
            ("Cooling System COP", results.cooling_system_cop, "-"),
            ("Cooling Electrical Energy per Thermal", results.cooling_elec_energy_per_thermal, "kWhe/kWhth"),
            ("Cooling CO₂ Emissions per Thermal", results.cooling_co2_emissions_per_thermal, "gCO₂/kWhth"),
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
        st.dataframe(df_cooling, use_container_width=True, hide_index=True)

# MAIN APPLICATION

def main():
    """
    Main function with case management integration 
    """
    # initialize session state and distributions
    initialize_session_state()
    initialize_default_distributions()
    
    # Check and initialize temporary variables
    temp_keys_exist = any(str(key).startswith('_temp_') for key in st.session_state.keys() if isinstance(key, str))
    if not temp_keys_exist:
        initialize_temp_variables_from_params()
    
    if 'sync_enabled' not in st.session_state:
        st.session_state.sync_enabled = True
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Quick Look'
    
    # get app state for case management
    app_state = get_app_state()
    
    # main page title
    st.title("Imperial ATES Calculator")
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
        
        # Operation buttons
        st.markdown("### Operations")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Calculate", type="primary", use_container_width=True):
                if perform_calculation():
                    st.rerun()
        
        with col2:
            if st.button("Reset", use_container_width=True):
                # Clear temporary variables
                temp_keys = [key for key in st.session_state.keys() 
                        if isinstance(key, str) and key.startswith('_temp_')]
                
                for key in temp_keys:
                    del st.session_state[key]
                
                # Clear calculation results and reset tracking
                result_keys = ['results', 'calculation_count', 'last_calculation_time', 'calculation_status']
                for key in result_keys:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Reset ATES parameters to default
                from tool.core.ates_calculator import ATESParameters
                st.session_state.ates_params = ATESParameters()
                
                # Reset calculation tracking
                st.session_state.calculation_count = 0
                st.session_state.calculation_status = 'not_started'
                
                # Reinitialize distributions
                initialize_default_distributions()
                
                # Mark case as modified
                mark_case_modified()
                
                st.success("Parameters reset to default values")
                st.rerun()
        
        with col3:
            if st.button("Validate", use_container_width=True):
                # Update parameters first, then validate
                update_all_parameters_from_temp()
                errors = validate_parameters()
                if errors:
                    for error in errors:
                        st.error(error)
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

if __name__ == "__main__":
    main()