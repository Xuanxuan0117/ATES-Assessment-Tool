"""
ATES Visualization 
provides classes and utility functions for visualizing and exporting
results from Monte Carlo simulations and sensitivity analyses.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from typing import Dict, List, Any, Optional, Literal, cast, Union
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')

def safe_float(value: Any) -> float:
    """
    Safely convert any value to float, handling all edge cases
    """
    try:
        if pd.isna(value):
            return 0.0
        if hasattr(value, 'item'):  #
            return float(value.item())
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if hasattr(value, '__float__'):
            return float(value)
        # complex type using str
        return float(str(value))
    except (ValueError, TypeError, AttributeError):
        return 0.0

def safe_abs(value: Any) -> float:
    """
    Safely get absolute value
    """
    try:
        # convert to safe float first, then get absolute value
        float_val = safe_float(value)
        return abs(float_val)
    except (ValueError, TypeError, AttributeError):
        return 0.0

def safe_int(value: Any) -> int:
    """
    safely convert any value to int
    """
    try:
        if pd.isna(value):
            return 0
        if hasattr(value, 'item'):
            return int(value.item())
        return int(float(value))
    except (ValueError, TypeError, AttributeError):
        return 0

class ATESVisualizer:
    """
    Main visualization class for Monte Carlo results and sensitivity analysis
    """
    
    def __init__(self, monte_carlo_results: Optional[pd.DataFrame] = None, 
             sensitivity_results: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Initialize the visualizer with Monte Carlo and sensitivity results
        
        Args:
            monte_carlo_results: df containing Monte Carlo simulation results
            sensitivity_results: dic containing sensitivity analysis results
        """
        self.monte_carlo_results = monte_carlo_results
        self.sensitivity_results = sensitivity_results
        
        if monte_carlo_results is not None:
            self.successful_results = monte_carlo_results[monte_carlo_results['success'] == True] if 'success' in monte_carlo_results.columns else monte_carlo_results
        else:
            self.successful_results = pd.DataFrame()
        
        # define parameter groups and display names
        self.parameter_groups = {
            'Heating System - Key Performance': {
                'heating_system_cop': 'System COP',
                'heating_annual_energy_building_GWhth': 'Annual Energy to Building (GWhth)',
                'heating_monthly_to_building': 'Average Monthly to Building (GWhth)',
                'heating_annual_elec_energy_GWhe': 'Annual Electricity (GWhe)',
                'heating_annual_energy_building_J': 'Annual Energy to Building (J)',
                'heating_annual_energy_building_kWhth': 'Annual Energy to Building (kWhth)',
                'heating_annual_elec_energy_J': 'Annual Electricity (J)',
                'heating_annual_elec_energy_MWhe': 'Annual Electricity (MWhe)',
                'heating_co2_emissions_per_thermal': 'CO₂ Emissions per Unit Thermal (gCO₂/kWhth)',
                'heating_elec_energy_per_thermal': 'Electrical Energy per Unit Thermal (kWhe/kWhth)'
            },
            
            'Heating System - Flow & Temperature': {
                'heating_total_flow_rate_ls': 'Total Flow Rate (l/s)',
                'heating_ave_production_temp': 'Average Production Temperature (°C)',
                'heating_total_flow_rate_m3hr': 'Total Flow Rate (m³/hr)',
                'heating_total_flow_rate_m3s': 'Total Flow Rate (m³/s)',
                'heating_ave_temp_change_across_HX': 'Average Temperature Change Across Heat Exchanger (°C)',
                'heating_temp_change_induced_HP': 'Temperature Change Induced by Heat Pump (°C)'
            },
            
            'Heating System - Energy & Storage': {
                'heating_total_energy_stored': 'Annual Energy Stored (J)',
                'heating_stored_energy_recovered': 'Annual Energy Recovered (J)',
                'heating_annual_energy_aquifer_GWhth': 'Annual Energy Produced from Aquifer (GWhth)',
                'heating_monthly_to_HX': 'Monthly Energy to Heat Exchanger (GWhth)',
                'heating_annual_energy_aquifer_J': 'Annual Energy Produced from Aquifer (J)',
                'heating_annual_energy_aquifer_kWhth': 'Annual Energy Produced from Aquifer (kWhth)'
            },
            
            'Heating System - Heat Pump & Electrical': {
                'heating_heat_pump_COP': 'Heat Pump COP',
                'heating_elec_energy_HP': 'Annual Energy to Heat Pump (J)',
                'heating_elec_energy_hydraulic_pumps': 'Annual Energy to Hydraulic Pump (J)',
                'heating_ehp': 'Heat Pump Factor (ehp)'
            },
            
            'Heating System - Power': {
                'heating_ave_power_to_HX_MW': 'Average Power to Heat Exchanger (MW)',
                'heating_ave_power_to_building_MW': 'Average Power to Building (MW)',
                'heating_ave_power_to_HX_W': 'Average Power to Heat Exchanger (W)',
                'heating_ave_power_to_building_W': 'Average Power to Building (W)'
            },
            
            'Cooling System - Key Performance': {
                'cooling_system_cop': 'System COP',
                'cooling_annual_energy_building_GWhth': 'Annual Energy to Building (GWhth)',
                'cooling_monthly_to_building': 'Average Monthly to Building (GWhth)',
                'cooling_annual_elec_energy_GWhe': 'Annual Electricity (GWhe)',
                'cooling_annual_energy_building_J': 'Annual Energy to Building (J)',
                'cooling_annual_energy_building_kWhth': 'Annual Energy to Building (kWhth)',
                'cooling_annual_elec_energy_J': 'Annual Electricity (J)',
                'cooling_annual_elec_energy_MWhe': 'Annual Electricity (MWhe)',
                'cooling_co2_emissions_per_thermal': 'CO₂ Emissions per Unit Thermal (gCO₂/kWhth)',
                'cooling_elec_energy_per_thermal': 'Electrical Energy per Unit Thermal (kWhe/kWhth)'
            },
            
            'Cooling System - Flow & Temperature': {
                'cooling_total_flow_rate_ls': 'Total Flow Rate (l/s)',
                'cooling_target_avg_flowrate_pd': 'Flow Rate per Borehole (m³/hr)',
                'cooling_ave_production_temp': 'Average Production Temperature (°C)',
                'cooling_total_flow_rate_m3hr': 'Total Flow Rate (m³/hr)',
                'cooling_total_flow_rate_m3s': 'Total Flow Rate (m³/s)',
                'cooling_ave_temp_change_across_HX': 'Average Temperature Change Across Heat Exchanger (°C)',
                'cooling_temp_change_induced_HP': 'Temperature Change Induced by Heat Pump (°C)'
            },
            
            'Cooling System - Energy & Storage': {
                'cooling_total_energy_stored': 'Annual Energy Stored (J)',
                'cooling_stored_energy_recovered': 'Annual Energy Recovered (J)',
                'cooling_annual_energy_aquifer_GWhth': 'Annual Energy Produced from Aquifer (GWhth)',
                'cooling_monthly_to_HX': 'Monthly Energy to Heat Exchanger (GWhth)',
                'cooling_annual_energy_aquifer_J': 'Annual Energy Produced from Aquifer (J)',
                'cooling_annual_energy_aquifer_kWhth': 'Annual Energy Produced from Aquifer (kWhth)'
            },
            
            'Cooling System - Heat Pump & Electrical': {
                'cooling_heat_pump_COP': 'Heat Pump COP',
                'cooling_elec_energy_HP': 'Annual Energy to Heat Pump (J)',
                'cooling_elec_energy_hydraulic_pumps': 'Annual Energy to Hydraulic Pump (J)',
                'cooling_ehp': 'Heat Pump Factor (ehp)'
            },
            
            'Cooling System - Power': {
                'cooling_ave_power_to_HX_MW': 'Average Power to Heat Exchanger (MW)',
                'cooling_ave_power_to_building_MW': 'Average Power to Building (MW)',
                'cooling_ave_power_to_HX_W': 'Average Power to Heat Exchanger (W)',
                'cooling_ave_power_to_building_W': 'Average Power to Building (W)'
            },
            
            'System Balance': {
                'energy_balance_ratio': 'Energy Balance Ratio (EBR)',
                'volume_balance_ratio': 'Volume Balance Ratio (VBR)'
            }
        }
        
        # Color scheme for different parameter groups
        self.group_colors = {
            'Heating System - Key Performance': '#FF6B6B',
            'Heating System - Energy & Storage': '#FF8E53', 
            'Heating System - Flow & Temperature': '#FF6B9D',
            'Heating System - Power': '#C44569',
            'Heating System - Heat Pump & Electrical': '#F8B500',
            'Cooling System - Key Performance': '#4ECDC4',
            'Cooling System - Energy & Storage': '#45B7D1',
            'Cooling System - Flow & Temperature': '#96CEB4', 
            'Cooling System - Power': '#FFEAA7',
            'Cooling System - Heat Pump & Electrical': '#DDA0DD',
            'System Balance & Overall': '#6C5CE7'
        }
    
    def render_distribution_plots(self):
        """Render frequency distribution plots for selected parameters"""
        st.subheader("Frequency Distributions")
        
        if self.monte_carlo_results is None:
            st.error("No Monte Carlo results available")
            st.info("Please run Monte Carlo analysis in the **Probabilistic Setup** screen first")
            return
        
        if len(self.successful_results) == 0:
            st.error("No successful simulation results to display")
            st.info(f"Total iterations: {len(self.monte_carlo_results)}, Successful: 0")
            return
        
        # Show simulation summary
        total_runs = len(self.monte_carlo_results)
        successful_runs = len(self.successful_results)
        success_rate = (successful_runs / total_runs) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Iterations", f"{total_runs:,}")
        with col2:
            st.metric("Successful", f"{successful_runs:,}")
        with col3:
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        st.markdown("---")
        
        # Parameter selection interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_group = st.selectbox(
                "Parameter Group",
                list(self.parameter_groups.keys()),
                key="dist_group"
            )
        
        with col2:
            plot_type = st.selectbox(
                "Plot Type",
                ["Histogram", "Box Plot", "Violin Plot", "Combined"],
                key="dist_plot_type"
            )
        
        # Get available parameters for selected group
        group_params = self.parameter_groups[selected_group]
        available_params = [param for param in group_params.keys() 
                            if param in self.successful_results.columns]
        
        if not available_params:
            st.warning(f"No data available for {selected_group} parameters")
            return
        
        # Parameter selection
        if 'Key Performance' in selected_group:
            default_count = 4
        elif 'Energy & Storage' in selected_group:
            default_count = 4
        elif 'Flow & Temperature' in selected_group:
            default_count = 3 if 'Cooling' in selected_group else 2
        elif 'Heat Pump & Electrical' in selected_group:
            default_count = 3
        elif 'Power' in selected_group:
            default_count = 2
        elif 'System Balance' in selected_group:
            default_count = 2
        else:
            default_count = 3

        default_selection = available_params[:default_count] if len(available_params) >= default_count else available_params
        selected_params = st.multiselect(
            "Select Parameters to Display",
            available_params,
            default=default_selection,
            format_func=lambda x: group_params[x],
            key="dist_params"
        )
        
        if not selected_params:
            st.warning("Please select at least one parameter to display")
            return
        
        # Generate plots based on selection
        if plot_type == "Histogram":
            self._plot_histograms(selected_params, group_params, selected_group)
        elif plot_type == "Box Plot":
            self._plot_box_plots(selected_params, group_params, selected_group)
        elif plot_type == "Violin Plot":
            self._plot_violin_plots(selected_params, group_params, selected_group)
        elif plot_type == "Combined":
            self._plot_combined_distributions(selected_params, group_params, selected_group)
        
        # Show summary statistics
        self._show_distribution_statistics(selected_params, group_params)
    
    def render_percentile_analysis(self):
        """
        Render percentile analysis tables and plots
        """
        st.subheader("Percentile Analysis")
        
        if self.monte_carlo_results is None:
            st.error("No Monte Carlo results available")
            st.info("Please run Monte Carlo analysis in the **Probabilistic Setup** screen first")
            return
        
        if len(self.successful_results) == 0:
            st.error("No successful simulation results to display")
            return
        
        # percentile configuration
        col1, col2 = st.columns(2)
        
        with col1:
            selected_group = st.selectbox(
                "Parameter Group",
                list(self.parameter_groups.keys()),
                key="perc_group"
            )
        
        with col2:
            percentile_type = st.selectbox(
                "Analysis Type",
                ["Detailed Percentiles", "Confidence Intervals"],
                key="perc_type"
            )
        
        group_params = self.parameter_groups[selected_group]
        available_params = [param for param in group_params.keys() 
                            if param in self.successful_results.columns]
        
        if not available_params:
            st.warning(f"No data available for {selected_group}")
            return
        elif percentile_type == "Detailed Percentiles":
            self._render_detailed_percentiles(available_params, group_params, selected_group)
        elif percentile_type == "Confidence Intervals":
            self._render_confidence_intervals(available_params, group_params, selected_group)
    
    def render_sensitivity_analysis(self):
        """
        Render sensitivity analysis results
        """
        st.subheader("Sensitivity Analysis")
        
        if not self.sensitivity_results:
            st.error("No sensitivity analysis results available")
            st.info("Sensitivity analysis is automatically calculated when you run Monte Carlo simulation")
            
            if self.monte_carlo_results is None:
                st.info("Go to **Probabilistic Setup** to run Monte Carlo analysis")
            else:
                st.warning("Try re-running the Monte Carlo simulation to generate sensitivity results")
            return
        
        # Show available output parameters
        output_params = list(self.sensitivity_results.keys())
        st.success(f"Sensitivity analysis available for {len(output_params)} output parameters")
        
        # interface controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_output = st.selectbox(
                "Output Parameter",
                output_params,
                format_func=lambda x: self._format_parameter_name(x),
                key="sens_output"
            )
        
        with col2:
            correlation_type = st.selectbox(
                "Correlation Type", 
                ["Pearson", "Spearman"],
                key="sens_corr_type"
            )
        
        with col3:
            n_top_params = st.selectbox(
                "Top Parameters",
                [5, 10, 15, 20],
                index=1,
                key="sens_n_params"
            )
        
        # display options
        show_options = st.columns(4)
        with show_options[0]:
            show_table = st.checkbox("Show Table", value=True)
        with show_options[1]:
            show_bar_chart = st.checkbox("Show Bar Chart", value=True)
        with show_options[2]:
            show_tornado = st.checkbox("Tornado Chart", value=False)
        with show_options[3]:
            show_importance = st.checkbox("Overall Ranking", value=False)
        
        # get sensitivity data for selected output
        sensitivity_df = self.sensitivity_results[selected_output]
        
        if show_table:
            self._show_sensitivity_table(sensitivity_df, selected_output)
        
        if show_bar_chart:
            self._plot_sensitivity_bar_chart(sensitivity_df, selected_output, correlation_type, n_top_params)
        
        if show_tornado:
            self._plot_tornado_chart(sensitivity_df, selected_output, correlation_type, n_top_params)
        
        if show_importance:
            self._plot_overall_parameter_importance()
    
    def render_correlation_matrix(self):
        """
        Render correlation matrix for output parameters
        """
        st.subheader("Output Parameter Correlations")
        
        if self.monte_carlo_results is None:
            st.error("No Monte Carlo results available")
            return
        
        if len(self.successful_results) == 0:
            st.error("No successful simulation results to display")
            return
        
        # select output parameters for correlation analysis
        numeric_cols = self.successful_results.select_dtypes(include=[np.number]).columns
        output_cols = [col for col in numeric_cols if not col.startswith('iteration')]
        
        if len(output_cols) < 2:
            st.warning("Need at least 2 numeric output parameters for correlation analysis")
            return
        
        # parameter selection
        selected_params = st.multiselect(
            "Select Parameters for Correlation Analysis",
            output_cols,
            default=output_cols[:8] if len(output_cols) >= 8 else output_cols,
            format_func=lambda x: self._format_parameter_name(x)
        )
        
        if len(selected_params) < 2:
            st.warning("Please select at least 2 parameters")
            return
        
        # correlation method
        corr_method = st.selectbox(
            "Correlation Method",
            ["pearson", "spearman", "kendall"],
            format_func=lambda x: x.title()
        )
        
        # calculate and display correlation matrix
        self._plot_correlation_matrix(selected_params, corr_method)

    def _plot_histograms(self, selected_params: List[str], group_params: Dict[str, str], group_name: str):
        """
        istogram plotting with data cleaning and dual Y-axis support
        """
        n_params = len(selected_params)
        
        # Display options
        with st.expander("Display Options", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                show_fit_line = st.checkbox("Show Normal Fit Line", value=False, key="fit_line_option")
            with col2:
                show_dual_axis = st.checkbox("Enable Dual Y-Axis", value=True, key="dual_axis_option")
            with col3:
                bins_count = st.slider("Number of Bins", 20, 100, 50, key="bins_option")
        
        if n_params == 1:
            param = selected_params[0]
            raw_data = self.successful_results[param]
            
            # Clean data 
            data = raw_data.dropna()
            data = data[np.isfinite(data)]
            
            if len(data) == 0:
                st.warning(f"No finite data available for {group_params[param]}")
                return
            
            # Check if we had infinite values and inform user
            infinite_count = len(raw_data) - len(raw_data.dropna()) - len(data)
            if infinite_count > 0:
                st.info(f"Note: {infinite_count} infinite values (direct mode) excluded from histogram")
            
            if show_dual_axis:
                # Create figure with secondary y-axis
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Calculate histogram data manually
                counts, bin_edges = np.histogram(data, bins=bins_count)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_width = bin_edges[1] - bin_edges[0]
                
                # Calculate probability
                probability = counts / len(data)
                
                # Add probability bars (primary Y-axis)
                fig.add_trace(
                    go.Bar(
                        x=bin_centers,
                        y=probability,
                        width=bin_width * 0.9,
                        name="Probability",
                        marker=dict(
                            color=self.group_colors.get(group_name, '#FF6B6B'),
                            line=dict(color='black', width=0.5)  
                        ),
                        opacity=0.7,
                        offsetgroup=0,
                        yaxis='y'
                    ),
                )
                
                # Add frequency bars (secondary Y-axis)
                fig.add_trace(
                    go.Bar(
                        x=bin_centers,
                        y=counts,
                        width=bin_width * 0.9,
                        name="Frequency",
                        marker=dict(
                            color='lightblue',
                            line=dict(color='black', width=0.5)  
                        ),
                        opacity=0.4,
                        offsetgroup=0,
                        yaxis='y2'
                    ),
                )
                
                # Add normal fit line if requested
                if show_fit_line and len(data) > 10:
                    x_range = np.linspace(data.min(), data.max(), 200)
                    mu, sigma = data.mean(), data.std()
                    if sigma > 0:
                        normal_density = stats.norm.pdf(x_range, mu, sigma)
                        normal_probability = normal_density * bin_width
                        
                        fig.add_trace(
                            go.Scatter(
                                x=x_range,
                                y=normal_probability,
                                mode='lines',
                                name='Normal Fit',
                                line=dict(color='orange', width=3, dash='dot'),
                                showlegend=True,
                                yaxis='y' 
                            )
                        )
                
                # Configure layout with proper dual Y-axis 
                fig.update_layout(
                        title={
                        'text': f"Distribution: {group_params[param]}",
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    height=500,
                    font=dict(color='black'),  
                    xaxis=dict(
                        title=group_params[param],
                        linecolor='black',     
                        tickcolor='black',    
                        ticks='outside',      
                        showline=True,         
                        mirror=True          
                    ),
                    yaxis=dict(
                        title="Probability", 
                        side="left",
                        range=[0, max(probability) * 1.1] if len(probability) > 0 else [0, 1],
                        linecolor='black',     
                        tickcolor='black',     
                        ticks='outside',      
                        showline=True,        
                        mirror=True           
                    ),
                    yaxis2=dict(
                        title="Frequency",
                        side="right",
                        overlaying="y",
                        range=[0, max(counts) * 1.1] if len(counts) > 0 else [0, 1],
                        linecolor='black',   
                        tickcolor='black',     
                        ticks='outside',     
                        showline=True        
                    ),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    barmode='overlay',
                    bargap=0,
                    bargroupgap=0,
                    plot_bgcolor='white',    
                    paper_bgcolor='white'     
                )
                
            else:
                # Single Y-axis mode 
                fig = go.Figure()
                fig.add_trace(
                    go.Histogram(
                        x=data,
                        nbinsx=bins_count,
                        name="Distribution",
                        marker=dict(
                            color=self.group_colors.get(group_name, '#FF6B6B'),
                            line=dict(color='black', width=0.5)  
                        ),
                        opacity=0.7,
                        histnorm='probability' 
                    )
                )
                
                # Add normal fit line if requested
                if show_fit_line and len(data) > 10:
                    x_range = np.linspace(data.min(), data.max(), 200)
                    mu, sigma = data.mean(), data.std()
                    if sigma > 0:
                        # Calculate bin width for proper scaling
                        _, bin_edges = np.histogram(data, bins=bins_count)
                        bin_width = bin_edges[1] - bin_edges[0]
                        
                        normal_density = stats.norm.pdf(x_range, mu, sigma)
                        normal_probability = normal_density * bin_width
                        
                        fig.add_trace(
                            go.Scatter(
                                x=x_range,
                                y=normal_probability,
                                mode='lines',
                                name='Normal Fit',
                                line=dict(color='orange', width=3, dash='dot'),
                                showlegend=True
                            )
                        )
                
                fig.update_layout(
                    title={
                        'text': f"Distribution: {group_params[param]}",
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    height=500,
                    font=dict(color='black'),  
                    xaxis=dict(
                        title=group_params[param],
                        linecolor='black',     
                        tickcolor='black',     
                        ticks='outside',      
                        showline=True,         
                        mirror=True            
                    ),
                    yaxis=dict(
                        title="Probability",
                        linecolor='black',     
                        tickcolor='black',     
                        ticks='outside',       
                        showline=True,        
                        mirror=True            
                    ),
                    plot_bgcolor='white',      
                    paper_bgcolor='white'      
                )
            
            # Add statistical lines
            mean_val = safe_float(data.mean())
            median_val = safe_float(data.median())
            
            fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                        annotation_text=f"Mean: {mean_val:.3f}")
            fig.add_vline(x=median_val, line_dash="dash", line_color="blue",
                        annotation_text=f"Median: {median_val:.3f}")
            
            st.plotly_chart(fig, width="stretch")
            
            # Show statistical summary
            st.markdown("### Statistical Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Count", f"{len(data):,}")
            with col2:
                st.metric("Mean", f"{mean_val:.4f}")
            with col3:
                st.metric("Median", f"{safe_float(data.median()):.4f}")
            with col4:
                st.metric("Std Dev", f"{safe_float(data.std()):.4f}")
        
        else:
            # Multiple parameters handling
            cols = min(2, n_params)
            
            for i in range(0, len(selected_params), cols):
                plot_cols = st.columns(cols)
                
                for j in range(cols):
                    if i + j < len(selected_params):
                        param = selected_params[i + j]
                        raw_data = self.successful_results[param]
                        
                        # Clean data for each parameter
                        data = raw_data.dropna()
                        data = data[np.isfinite(data)]
                        
                        if len(data) == 0:
                            with plot_cols[j]:
                                st.warning(f"No finite data for {group_params[param]}")
                            continue
                        
                        # Check for infinite values
                        infinite_count = len(raw_data) - len(raw_data.dropna()) - len(data)
                        
                        if show_dual_axis:
                            # Create subplot with dual Y-axis
                            fig = make_subplots(specs=[[{"secondary_y": True}]])
                            
                            # Calculate histogram data
                            counts, bin_edges = np.histogram(data, bins=bins_count)
                            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                            bin_width = bin_edges[1] - bin_edges[0]
                            probability = counts / len(data)
                            
                            # Probability bars (primary Y-axis)
                            fig.add_trace(
                                go.Bar(
                                    x=bin_centers,
                                    y=probability,
                                    width=bin_width * 0.9,
                                    name="Probability",
                                    marker=dict(
                                        color=self.group_colors.get(group_name, '#FF6B6B'),
                                        line=dict(color='black', width=0.5)  
                                    ),
                                    opacity=0.7,
                                    offsetgroup=0,
                                    yaxis='y'
                                )
                            )
                            
                            # Frequency bars (secondary Y-axis)
                            fig.add_trace(
                                go.Bar(
                                    x=bin_centers,
                                    y=counts,
                                    width=bin_width * 0.9,
                                    name="Frequency",
                                    marker=dict(
                                        color='lightblue',
                                        line=dict(color='black', width=0.5),  
                                        opacity=0.4
                                    ),
                                    offsetgroup=0,
                                    yaxis='y2'
                                )
                            )
                            
                            # Configure layout 
                            fig.update_layout(
                                title={
                                    'text': group_params[param],
                                    'x': 0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top'
                                },
                                height=350,
                                title_x=0.5,
                                title_font_size=12,
                                showlegend=False,
                                margin=dict(l=40, r=40, t=40, b=40),
                                barmode='overlay',
                                bargap=0,
                                bargroupgap=0,
                                font=dict(color='black'),  
                                xaxis=dict(
                                    title="Value",
                                    linecolor='black',     
                                    tickcolor='black',   
                                    ticks='outside',       
                                    showline=True,         
                                    mirror=True            
                                ),
                                yaxis=dict(
                                    title="Probability", 
                                    side="left",
                                    range=[0, max(probability) * 1.1] if len(probability) > 0 else [0, 1],
                                    linecolor='black',     
                                    tickcolor='black',   
                                    ticks='outside',      
                                    showline=True,         
                                    mirror=True           
                                ),
                                yaxis2=dict(
                                    title="Freq", 
                                    side="right", 
                                    overlaying="y",
                                    range=[0, max(counts) * 1.1] if len(counts) > 0 else [0, 1],
                                    linecolor='black',     
                                    tickcolor='black',    
                                    ticks='outside',      
                                    showline=True          
                                ),
                                plot_bgcolor='white',      
                                paper_bgcolor='white'     
                            )
                            
                            # Add normal fit if requested
                            if show_fit_line and len(data) > 10:
                                x_range = np.linspace(data.min(), data.max(), 100)
                                mu, sigma = data.mean(), data.std()
                                if sigma > 0:
                                    normal_density = stats.norm.pdf(x_range, mu, sigma)
                                    normal_probability = normal_density * bin_width
                                    
                                    fig.add_trace(
                                        go.Scatter(
                                            x=x_range, 
                                            y=normal_probability,
                                            mode='lines', 
                                            name='Fit',
                                            line=dict(color='orange', width=2, dash='dot'),
                                            showlegend=False,
                                            yaxis='y'
                                        )
                                    )
                            
                        else:
                            # Single Y-axis mode
                            fig = go.Figure()
                            fig.add_trace(
                                go.Histogram(
                                    x=data,
                                    nbinsx=bins_count,
                                    marker=dict(
                                        color=self.group_colors.get(group_name, '#FF6B6B'),
                                        line=dict(color='black', width=0.5)  # 黑色边框
                                    ),
                                    opacity=0.7,
                                    histnorm='probability'
                                )
                            )
                            
                            # Add normal fit
                            if show_fit_line and len(data) > 10:
                                x_range = np.linspace(data.min(), data.max(), 100)
                                mu, sigma = data.mean(), data.std()
                                if sigma > 0:
                                    _, bin_edges = np.histogram(data, bins=bins_count)
                                    bin_width = bin_edges[1] - bin_edges[0]
                                    
                                    normal_density = stats.norm.pdf(x_range, mu, sigma)
                                    normal_probability = normal_density * bin_width
                                    
                                    fig.add_trace(
                                        go.Scatter(
                                            x=x_range, y=normal_probability,
                                            mode='lines', name='Fit',
                                            line=dict(color='orange', width=2, dash='dot'),
                                            showlegend=False
                                        )
                                    )
                            
                            fig.update_layout(
                                title=group_params[param],
                                height=350,
                                title_x=0.5,
                                title_font_size=12,
                                showlegend=False,
                                margin=dict(l=40, r=40, t=40, b=40),
                                font=dict(color='black'),  
                                xaxis=dict(
                                    title="Value",
                                    linecolor='black',     
                                    tickcolor='black',     
                                    ticks='outside',      
                                    showline=True,         
                                    mirror=True            
                                ),
                                yaxis=dict(
                                    title="Probability",
                                    linecolor='black',     
                                    tickcolor='black',     
                                    ticks='outside',      
                                    showline=True,        
                                    mirror=True            
                                ),
                                plot_bgcolor='white',     
                                paper_bgcolor='white'     
                            )
                        
                        # Add mean line
                        mean_val = safe_float(data.mean())
                        fig.add_vline(x=mean_val, line_dash="dash", line_color="red", line_width=1)
                        
                        with plot_cols[j]:
                            st.plotly_chart(fig, width="stretch")
                            
                            # Statistics below chart
                            col_stat1, col_stat2 = st.columns(2)
                            with col_stat1:
                                st.caption(f"μ: {mean_val:.3f}")
                            with col_stat2:
                                st.caption(f"σ: {safe_float(data.std()):.3f}")
                            
                            # Show infinite count if any
                            if infinite_count > 0:
                                st.caption(f"⚠ {infinite_count} infinite values excluded")

        
    def _plot_box_plots(self, selected_params: List[str], group_params: Dict[str, str], group_name: str):
        """
        Plot box plots for selected parameters
        """
        # prepare data 
        plot_data = []
        for param in selected_params:
            data = self.successful_results[param].dropna()
            data = data[np.isfinite(data)]
            for value in data:
                plot_data.append({
                    'Parameter': group_params[param],
                    'Value': safe_float(value),  # Safe conversion
                    'param_key': param
                })
        
        if not plot_data:
            st.warning("No data available for box plots")
            return
        
        plot_df = pd.DataFrame(plot_data)
        
        fig = px.box(
            plot_df,
            x='Parameter',
            y='Value',
            title=f"{group_name} - Box Plot Analysis",
            color='Parameter',
            color_discrete_sequence=[self.group_colors[group_name]] * len(selected_params)
        )
        
        fig.update_layout(
            title_x=0.5,
            height=500,
            showlegend=False,
            xaxis={'tickangle': 45}
        )
        
        st.plotly_chart(fig, width="stretch")
    
    def _plot_violin_plots(self, selected_params: List[str], group_params: Dict[str, str], group_name: str):
        """
        Plot violin plots for selected parameters
        """
        # Prepare data for violin plots
        plot_data = []
        for param in selected_params:
            data = self.successful_results[param].dropna()
            data = data[np.isfinite(data)]
            for value in data:
                plot_data.append({
                    'Parameter': group_params[param],
                    'Value': safe_float(value)  # Safe conversion
                })
        
        if not plot_data:
            st.warning("No data available for violin plots")
            return
        
        plot_df = pd.DataFrame(plot_data)
        
        fig = px.violin(
            plot_df,
            x='Parameter',
            y='Value',
            title=f"{group_name} - Violin Plot Analysis",
            box=True,
            color='Parameter',
            color_discrete_sequence=[self.group_colors[group_name]] * len(selected_params)
        )
        
        fig.update_layout(
            title_x=0.5,
            height=500,
            showlegend=False,
            xaxis={'tickangle': 45}
        )
        
        st.plotly_chart(fig, width="stretch")

    def _plot_combined_distributions(self, selected_params: List[str], group_params: Dict[str, str], group_name: str):
        """Plot combined histogram, box plot, and Q-Q plot for each parameter"""
        for param in selected_params:
            data = self.successful_results[param].dropna()
            data = data[np.isfinite(data)]
            if len(data) == 0:
                continue
            
            # Create subplots: histogram + box plot + Q-Q Plot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=["Distribution", "Box Plot", "Q-Q Plot"],
                specs=[[{"secondary_y": False}, {"rowspan": 2}],
                       [{"secondary_y": False}, None]], # None for empty bottom right
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=data, 
                    nbinsx=30, 
                    name="Frequency",
                    marker_color=self.group_colors[group_name],
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(
                    y=data, 
                    name="Distribution",
                    marker_color=self.group_colors[group_name]
                ),
                row=1, col=2
            )
            
            # Q-Q plot
            sorted_data = np.sort(data)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sorted_data,
                    mode='markers',
                    name="Q-Q Plot",
                    marker=dict(color=self.group_colors[group_name])
                ),
                row=2, col=1
            )
            
            # Add reference line for Q-Q plot with safe conversions
            min_val = safe_float(np.min(theoretical_quantiles))
            max_val = safe_float(np.max(theoretical_quantiles))
            q01_val = safe_float(data.quantile(0.01))
            q99_val = safe_float(data.quantile(0.99))
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[q01_val, q99_val],
                    mode='lines',
                    name="Reference",
                    line=dict(dash='dash', color='red')
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title_text=f"{group_params[param]} - Comprehensive Analysis",
                title_x=0.5,
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig, width="stretch")

    def _show_distribution_statistics(self, selected_params: List[str], group_params: Dict[str, str]):
        """Show summary statistics for selected parameters"""
        st.markdown("### Summary Statistics")
        
        stats_data = []
        for param in selected_params:
            data = self.successful_results[param].dropna()
            data = data[np.isfinite(data)]
            if len(data) > 0:
                stats_data.append({
                    'Parameter': group_params[param],
                    'Count': len(data),
                    'Mean': safe_float(data.mean()),
                    'Std Dev': safe_float(data.std()),
                    'Min': safe_float(data.min()),
                    'Q25': safe_float(data.quantile(0.25)),
                    'Median': safe_float(data.median()),
                    'Q75': safe_float(data.quantile(0.75)),
                    'Max': safe_float(data.max())
                })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            # Format numerical columns
            numeric_cols = ['Count', 'Mean', 'Std Dev', 'Min', 'Q25', 'Median', 'Q75', 'Max']
            for col in numeric_cols:
                if col == 'Count':
                    stats_df[col] = stats_df[col].astype(int)
                else:
                    stats_df[col] = stats_df[col].round(4)
            
            st.dataframe(stats_df, width="stretch", hide_index=True)

    def _render_risk_analysis(self, available_params: List[str], group_params: Dict[str, str], group_name: str):
        """Render risk analysis with performance thresholds"""
        st.markdown("### Risk Analysis")
        
        # Define performance thresholds based on parameter type
        thresholds = {
            'heating_system_cop': {'excellent': 4.0, 'good': 3.0, 'acceptable': 2.0},
            'cooling_system_cop': {'excellent': 5.0, 'good': 3.5, 'acceptable': 2.5},
            'energy_balance_ratio': {'excellent': 0.05, 'good': 0.1, 'acceptable': 0.2},
            'heating_annual_energy_building_GWhth': {'excellent': 8.0, 'good': 4.0, 'acceptable': 2.0},
            'cooling_annual_energy_building_GWhth': {'excellent': 5.0, 'good': 2.5, 'acceptable': 1.0}
        }
        
        risk_data = []
        for param in available_params:
            if param in thresholds:
                data = self.successful_results[param].dropna()
                data = data[np.isfinite(data)]
                if len(data) > 0:
                    threshold = thresholds[param]
                    
                    if param == 'energy_balance_ratio':
                        abs_data = data.abs()
                        excellent = safe_float((abs_data <= threshold['excellent']).mean()) * 100
                        good = safe_float((abs_data <= threshold['good']).mean()) * 100
                        acceptable = safe_float((abs_data <= threshold['acceptable']).mean()) * 100
                    else:
                        # For most parameters, higher is better
                        excellent = safe_float((data >= threshold['excellent']).mean()) * 100
                        good = safe_float((data >= threshold['good']).mean()) * 100
                        acceptable = safe_float((data >= threshold['acceptable']).mean()) * 100
                    
                    risk_data.append({
                        'Parameter': group_params[param],
                        'Excellent (%)': excellent,
                        'Good (%)': good,
                        'Acceptable (%)': acceptable,
                        'Poor (%)': 100 - acceptable
                    })
        
        if risk_data:
            risk_df = pd.DataFrame(risk_data)
            
            # Display risk table
            st.dataframe(risk_df.round(1), width="stretch", hide_index=True)
            
            # Risk visualization
            fig = go.Figure()
            
            categories = ['Excellent', 'Good', 'Acceptable', 'Poor']
            colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']
            
            for i, category in enumerate(categories):
                col_name = f"{category} (%)"
                if col_name in risk_df.columns:
                    fig.add_trace(go.Bar(
                        x=risk_df['Parameter'],
                        y=risk_df[col_name],
                        name=category,
                        marker_color=colors[i]
                    ))
            
            fig.update_layout(
                title="Performance Risk Analysis",
                title_x=0.5,
                barmode='stack',
                height=500,
                xaxis_title="Parameters",
                yaxis_title="Probability (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis={'tickangle': 45}
            )
            
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Risk analysis not available for selected parameters")

    def _render_detailed_percentiles(self, available_params: List[str], group_params: Dict[str, str], group_name: str):
        """
        Render detailed percentile analysis
        """
        st.markdown("### Detailed Percentile Analysis")
        
        # Percentile selection 
        percentile_options = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        selected_percentiles = st.multiselect(
            "Select Percentiles to Display",
            percentile_options,
            default=[0, 10, 30, 50, 70, 90, 100],
            key="detailed_percentiles"
        )
        
        if not selected_percentiles:
            st.warning("Please select at least one percentile")
            return
        
        # Calculate percentiles
        percentile_data = []
        for param in available_params:
            data = self.successful_results[param].dropna()
            finite_data = data[np.isfinite(data)]
            
            if len(finite_data) > 0: 
                row = {
                    'Parameter': group_params[param],
                    'Mean': safe_float(finite_data.mean()),
                    'Std': safe_float(finite_data.std())
                }
                
                for p in selected_percentiles:
                    row[f'P{p}'] = safe_float(finite_data.quantile(p/100))
                
                infinite_count = len(data) - len(finite_data)
                if infinite_count > 0:
                    row['Parameter'] += f" ({infinite_count} direct mode)"
                    
                percentile_data.append(row)
        
        if percentile_data:
            percentile_df = pd.DataFrame(percentile_data)
            
            # Format numerical columns
            numeric_cols = [col for col in percentile_df.columns if col != 'Parameter']
            for col in numeric_cols:
                percentile_df[col] = percentile_df[col].round(4)
            
            # Display table
            st.dataframe(percentile_df, width="stretch", hide_index=True)
            
            # Percentile chart
            self._plot_percentile_chart(percentile_df, selected_percentiles, group_name)

    def _render_confidence_intervals(self, available_params: List[str], group_params: Dict[str, str], group_name: str):
        """
        Render confidence interval analysis
        """
        st.markdown("### Confidence Intervals")
        
        # Confidence level selection
        confidence_level = st.selectbox(
            "Confidence Level",
            [90, 95, 99],
            index=1,
            format_func=lambda x: f"{x}%"
        )
        
        alpha = (100 - confidence_level) / 100
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_data = []
        for param in available_params:
            data = self.successful_results[param].dropna()
            data = data[np.isfinite(data)]
            if len(data) > 0:
                mean = safe_float(data.mean())
                std = safe_float(data.std())
                n = safe_int(len(data))
                
                # Bootstrap confidence interval (using quantiles as an approximation)
                lower_ci = safe_float(data.quantile(lower_percentile / 100))
                upper_ci = safe_float(data.quantile(upper_percentile / 100))
                
                ci_data.append({
                    'Parameter': group_params[param],
                    'Mean': mean,
                    'Std': std,
                    'Sample Size': n,
                    f'CI Lower ({confidence_level}%)': lower_ci,
                    f'CI Upper ({confidence_level}%)': upper_ci,
                    'CI Width': safe_float(upper_ci - lower_ci)
                })
        
        if ci_data:
            ci_df = pd.DataFrame(ci_data)
            
            # Format numerical columns
            numeric_cols = [col for col in ci_df.columns if col not in ['Parameter', 'Sample Size']]
            for col in numeric_cols:
                ci_df[col] = ci_df[col].round(4)
            
            # Display table
            st.dataframe(ci_df, width="stretch", hide_index=True)
            
            # Confidence interval plot
            self._plot_confidence_intervals(ci_df, confidence_level, group_name)

    def _plot_percentile_chart(self, percentile_df: pd.DataFrame, selected_percentiles: List[int], group_name: str):
        """
        Plot percentile chart
        """
        fig = go.Figure()
        
        percentile_cols = [f'P{p}' for p in selected_percentiles]
        
        for i, row in percentile_df.iterrows():
            param_name = row['Parameter']
            y_values = [safe_float(row[col]) for col in percentile_cols if col in row]
            
            fig.add_trace(go.Scatter(
                x=selected_percentiles,
                y=y_values,
                mode='lines+markers',
                name=param_name,
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=f"{group_name} - Percentile Analysis",
            title_x=0.5,
            xaxis_title="Percentile",
            yaxis_title="Value",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, width="stretch")

    def _plot_confidence_intervals(self, ci_df: pd.DataFrame, confidence_level: int, group_name: str):
        """
        Plot confidence intervals
        """
        fig = go.Figure()
        
        for i, row in ci_df.iterrows():
            param_name = row['Parameter']
            mean_val = safe_float(row['Mean'])
            lower_ci = safe_float(row[f'CI Lower ({confidence_level}%)'])
            upper_ci = safe_float(row[f'CI Upper ({confidence_level}%)'])
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=[param_name, param_name],
                y=[lower_ci, upper_ci],
                mode='lines',
                line=dict(width=8, color='lightgray'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Add mean point
            fig.add_trace(go.Scatter(
                x=[param_name],
                y=[mean_val],
                mode='markers',
                marker=dict(size=12, color=self.group_colors[group_name]),
                name=param_name,
                showlegend=False
            ))
        
        fig.update_layout(
            title=f"{group_name} - {confidence_level}% Confidence Intervals",
            title_x=0.5,
            height=500,
            xaxis_title="Parameters",
            yaxis_title="Value",
            xaxis={'tickangle': 45}
        )
        
        st.plotly_chart(fig, width="stretch")

    def _show_sensitivity_table(self, sensitivity_df: pd.DataFrame, selected_output: str):
        """
        Show sensitivity analysis table
        """
        st.markdown(f"### Sensitivity Table: {self._format_parameter_name(selected_output)}")
        
        # Format the table for display
        display_df = sensitivity_df.copy()
        display_df['Input_Parameter'] = display_df['Input_Parameter'].apply(self._format_parameter_name)
        
        # Round correlation values
        numeric_cols = ['Pearson_Correlation', 'Spearman_Correlation', 'Abs_Pearson', 'Abs_Spearman']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(4)
        
        # Rename columns for better display
        display_df = display_df.rename(columns={
            'Input_Parameter': 'Input Parameter',
            'Pearson_Correlation': 'Pearson Correlation',
            'Spearman_Correlation': 'Spearman Correlation',
            'Abs_Pearson': 'Abs. Pearson',
            'Abs_Spearman': 'Abs. Spearman'
        })
        
        st.dataframe(display_df, width="stretch", hide_index=True)

    def _plot_sensitivity_bar_chart(self, sensitivity_df: pd.DataFrame, selected_output: str, 
                                 correlation_type: str, n_top_params: int):
        """
        Plot sensitivity analysis bar chart
        """
        # Get top N parameters
        top_params = sensitivity_df.head(n_top_params).copy()
        
        # select correlation column
        corr_col = f"{correlation_type}_Correlation"
        abs_corr_col = f"Abs_{correlation_type}"
        
        if corr_col not in top_params.columns:
            st.error(f"{correlation_type} correlation data not available")
            return
        
        # format parameter names
        top_params['Formatted_Name'] = top_params['Input_Parameter'].apply(self._format_parameter_name)
        
        # Create bar chart
        fig = px.bar(
            top_params,
            x='Formatted_Name',
            y=abs_corr_col,
            title=f"Top {n_top_params} Most Influential Parameters - {self._format_parameter_name(selected_output)}",
            labels={
                'Formatted_Name': 'Input Parameters',
                abs_corr_col: f'Absolute {correlation_type} Correlation'
            },
            color=abs_corr_col,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            title_x=0.5,
            height=500,
            showlegend=False,
            xaxis={'tickangle': 45}
        )
        
        # Add correlation values as text on bars
        fig.update_traces(
            text=top_params[corr_col].round(3),
            textposition='outside'
        )
        
        st.plotly_chart(fig, width="stretch")

    def _plot_tornado_chart(self, sensitivity_df: pd.DataFrame, selected_output: str, 
                         correlation_type: str, n_top_params: int):
        """Plot tornado chart for sensitivity analysis"""
        # Get top N parameters
        top_params = sensitivity_df.head(n_top_params).copy()
        
        corr_col = f"{correlation_type}_Correlation"
        
        if corr_col not in top_params.columns:
            st.error(f"{correlation_type} correlation data not available")
            return
        
        # Format parameter names and sort by absolute correlation
        top_params['Formatted_Name'] = top_params['Input_Parameter'].apply(self._format_parameter_name)
        top_params = top_params.sort_values(f"Abs_{correlation_type}", ascending=True)
        
        # Create tornado chart
        fig = go.Figure()
        
        # positive correlations
        positive_mask = top_params[corr_col] >= 0
        if positive_mask.any():
            fig.add_trace(go.Bar(
                x=top_params[positive_mask][corr_col],
                y=top_params[positive_mask]['Formatted_Name'],
                orientation='h',
                name='Positive Correlation',
                marker_color='green',
                opacity=0.7
            ))
        
        # negative correlations
        negative_mask = top_params[corr_col] < 0
        if negative_mask.any():
            fig.add_trace(go.Bar(
                x=top_params[negative_mask][corr_col],
                y=top_params[negative_mask]['Formatted_Name'],
                orientation='h',
                name='Negative Correlation',
                marker_color='red',
                opacity=0.7
            ))
        
        fig.update_layout(
            title=f"Tornado Chart - {self._format_parameter_name(selected_output)}",
            title_x=0.5,
            height=max(400, n_top_params * 30),
            xaxis_title=f"{correlation_type} Correlation",
            yaxis_title="Input Parameters",
            barmode='relative'
        )
        
        # Add vertical line at x=0
        fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
        
        st.plotly_chart(fig, width="stretch")

    def _plot_overall_parameter_importance(self):
        """
        Plot overall parameter importance ranking
        """
        if not self.sensitivity_results:
            st.warning("No sensitivity results available")
            return
        
        st.markdown("### Overall Parameter Importance Ranking")
        
        # Calculate importance scores across all outputs
        importance_scores = {}
        
        for output_param, sensitivity_df in self.sensitivity_results.items():
            for _, row in sensitivity_df.iterrows():
                input_param = row['Input_Parameter']
                abs_corr = safe_float(row['Abs_Pearson'])
                
                if input_param not in importance_scores:
                    importance_scores[input_param] = []
                importance_scores[input_param].append(abs_corr)
        
        # Calculate aggregate metrics
        ranking_data = []
        for input_param, scores in importance_scores.items():
            ranking_data.append({
                'Parameter': self._format_parameter_name(input_param),
                'Mean_Importance': np.mean(scores),
                'Max_Importance': np.max(scores),
                'Min_Importance': np.min(scores),
                'Std_Importance': np.std(scores),
                'Number_of_Outputs': len(scores)
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('Mean_Importance', ascending=True)
        
        # Display top parameters
        n_display = st.slider("Number of parameters to display", 5, min(20, len(ranking_df)), 10)
        top_ranking = ranking_df.tail(n_display)
        
        # Create horizontal bar chart
        fig = px.bar(
            top_ranking,
            x='Mean_Importance',
            y='Parameter',
            orientation='h',
            title="Overall Parameter Importance Ranking (Mean Absolute Correlation)",
            labels={
                'Mean_Importance': 'Mean Absolute Correlation',
                'Parameter': 'Input Parameters'
            },
            color='Mean_Importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            title_x=0.5,
            height=max(400, n_display * 30),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, width="stretch")
        
        # Show detailed ranking table
        if st.checkbox("Show Detailed Ranking Table"):
            display_ranking = ranking_df.sort_values('Mean_Importance', ascending=False)
            
            # round numerical columns
            numeric_cols = ['Mean_Importance', 'Max_Importance', 'Min_Importance', 'Std_Importance']
            for col in numeric_cols:
                display_ranking[col] = display_ranking[col].round(4)
            
            st.dataframe(display_ranking, width="stretch", hide_index=True)

    def _plot_correlation_matrix(self, selected_params: List[str], corr_method: str):
        """
        Plot correlation matrix for selected parameters
        """
        # Calculate correlation matrix
        try:
            correlation_data = self.successful_results[selected_params]
            correlation_data = correlation_data.replace([np.inf, -np.inf], np.nan).dropna()
            corr_method_typed = cast(Literal["pearson", "spearman", "kendall"], corr_method)
            correlation_matrix = correlation_data.corr(method=corr_method_typed)
            
            # create heatmap
            fig = px.imshow(
                correlation_matrix,
                title=f"Output Parameter Correlation Matrix ({corr_method.title()})",
                color_continuous_scale="RdBu_r",
                aspect="auto",
                zmin=-1,
                zmax=1
            )
            
            # format labels
            formatted_labels = [self._format_parameter_name(col) for col in correlation_matrix.columns]
            fig.update_layout(
                title_x=0.5,
                height=600,
                xaxis_title="Parameters",
                yaxis_title="Parameters",
                xaxis=dict(
                    ticktext=formatted_labels, 
                    tickvals=list(range(len(formatted_labels))),
                    tickangle=45
                ),
                yaxis=dict(
                    ticktext=formatted_labels, 
                    tickvals=list(range(len(formatted_labels)))
                )
            )
            
            st.plotly_chart(fig, width="stretch")
            
            # Show correlation statistics
            if st.checkbox("Show Correlation Statistics"):
                # find strongest correlations
                corr_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        param1 = correlation_matrix.columns[i]
                        param2 = correlation_matrix.columns[j]
                        corr_value = correlation_matrix.iloc[i, j]
                        
                        corr_pairs.append({
                            'Parameter 1': self._format_parameter_name(param1),
                            'Parameter 2': self._format_parameter_name(param2),
                            'Correlation': safe_float(corr_value),
                            'Abs Correlation': safe_abs(corr_value)
                        })
                
                corr_pairs_df = pd.DataFrame(corr_pairs)
                corr_pairs_df = corr_pairs_df.sort_values('Abs Correlation', ascending=False)
                corr_pairs_df['Correlation'] = corr_pairs_df['Correlation'].round(4)
                corr_pairs_df['Abs Correlation'] = corr_pairs_df['Abs Correlation'].round(4)
                
                st.markdown("**Strongest Correlations:**")
                st.dataframe(corr_pairs_df.head(10), width="stretch", hide_index=True)
        
        except Exception as e:
            st.error(f"Error calculating correlation matrix: {str(e)}")

    def _format_parameter_name(self, param_name: str) -> str:
        """
        Format parameter names for display
        """
        name_mapping = {
            # Input parameters 
            'aquifer_temp': 'Aquifer Temperature',
            'water_density': 'Water Density',
            'water_specific_heat_capacity': 'Water Specific Heat Capacity',  
            'thermal_recovery_factor': 'Thermal Recovery Factor',
            'heating_target_avg_flowrate_pd': 'Heating Target Average Flow Rate per Doublet', 
            'tolerance_in_energy_balance': 'Energy Balance Tolerance',  
            'heating_number_of_doublets': 'Number of Doublets',  
            'heating_months': 'Heating Months',
            'cooling_months': 'Cooling Months',
            'pump_energy_density': 'Pump Energy Density',
            'heating_ave_injection_temp': 'Heating Average Injection Temperature',
            'heating_temp_to_building': 'Heating Temperature to Building', 
            'cop_param_a': 'COP Parameter A',
            'cop_param_b': 'COP Parameter B',
            'cop_param_c': 'COP Parameter C',
            'cop_param_d': 'COP Parameter D',
            'carbon_intensity': 'Carbon Intensity',
            'cooling_ave_injection_temp': 'Cooling Average Injection Temperature',
            'cooling_temp_to_building': 'Cooling Temperature to Building',
            
            # Output parameters - Heating
            'heating_system_cop': 'Heating System COP',
            'heating_annual_energy_building_GWhth': 'Heating Annual Energy to Building (GWhth)',
            'heating_annual_elec_energy_GWhe': 'Heating Annual Electrical Energy (GWhe)',
            'heating_co2_emissions_per_thermal': 'Heating CO₂ Emissions per Thermal',
            'heating_ave_power_to_building_MW': 'Heating Average Power to Building (MW)',
            'heating_ave_production_temp': 'Heating Average Production Temperature',
            'heating_total_energy_stored': 'Heating Total Energy Stored',
            'heating_stored_energy_recovered': 'Heating Stored Energy Recovered',
            'heating_annual_energy_aquifer_J': 'Heating Annual Energy from Aquifer (J)',
            'heating_annual_energy_aquifer_kWhth': 'Heating Annual Energy from Aquifer (kWhth)',
            'heating_annual_energy_aquifer_GWhth': 'Heating Annual Energy from Aquifer (GWhth)',
            'heating_annual_energy_building_J': 'Heating Annual Energy to Building (J)',
            'heating_annual_energy_building_kWhth': 'Heating Annual Energy to Building (kWhth)',
            'heating_elec_energy_per_thermal': 'Heating Electrical Energy per Thermal',
            'heating_total_flow_rate_m3hr': 'Heating Total Flow Rate (m³/hr)',
            'heating_total_flow_rate_ls': 'Heating Total Flow Rate (l/s)',
            'heating_total_flow_rate_m3s': 'Heating Total Flow Rate (m³/s)',
            'heating_ave_temp_change_across_HX': 'Heating Average Temperature Change Across HX',
            'heating_temp_change_induced_HP': 'Heating Temperature Change Induced by HP',
            'heating_heat_pump_COP': 'Heating Heat Pump COP',
            'heating_ehp': 'Heating Heat Pump Factor (ehp)',
            'heating_ave_power_to_HX_W': 'Heating Average Power to HX (W)',
            'heating_ave_power_to_HX_MW': 'Heating Average Power to HX (MW)',
            'heating_ave_power_to_building_W': 'Heating Average Power to Building (W)',
            'heating_monthly_to_HX': 'Heating Monthly Energy to HX',
            'heating_monthly_to_building': 'Heating Monthly Energy to Building',
            'heating_elec_energy_hydraulic_pumps': 'Heating Electrical Energy to Hydraulic Pumps',
            'heating_elec_energy_HP': 'Heating Electrical Energy to Heat Pump',
            'heating_annual_elec_energy_J': 'Heating Annual Electrical Energy (J)',
            'heating_annual_elec_energy_MWhe': 'Heating Annual Electrical Energy (MWhe)',
            
            # Output parameters - Cooling
            'cooling_system_cop': 'Cooling System COP',
            'cooling_annual_energy_building_GWhth': 'Cooling Annual Energy to Building (GWhth)',
            'cooling_annual_elec_energy_GWhe': 'Cooling Annual Electrical Energy (GWhe)',
            'cooling_co2_emissions_per_thermal': 'Cooling CO₂ Emissions per Thermal',
            'cooling_ave_power_to_building_MW': 'Cooling Average Power to Building (MW)',
            'cooling_ave_production_temp': 'Cooling Average Production Temperature',
            'cooling_total_energy_stored': 'Cooling Total Energy Stored',
            'cooling_stored_energy_recovered': 'Cooling Stored Energy Recovered',
            'cooling_annual_energy_aquifer_J': 'Cooling Annual Energy from Aquifer (J)',
            'cooling_annual_energy_aquifer_kWhth': 'Cooling Annual Energy from Aquifer (kWhth)',
            'cooling_annual_energy_aquifer_GWhth': 'Cooling Annual Energy from Aquifer (GWhth)',
            'cooling_annual_energy_building_J': 'Cooling Annual Energy to Building (J)',
            'cooling_annual_energy_building_kWhth': 'Cooling Annual Energy to Building (kWhth)',
            'cooling_elec_energy_per_thermal': 'Cooling Electrical Energy per Thermal',
            'cooling_total_flow_rate_m3hr': 'Cooling Total Flow Rate (m³/hr)',
            'cooling_total_flow_rate_ls': 'Cooling Total Flow Rate (l/s)',
            'cooling_total_flow_rate_m3s': 'Cooling Total Flow Rate (m³/s)',
            'cooling_ave_temp_change_across_HX': 'Cooling Average Temperature Change Across HX',
            'cooling_temp_change_induced_HP': 'Cooling Temperature Change Induced by HP',
            'cooling_heat_pump_COP': 'Cooling Heat Pump COP',
            'cooling_ehp': 'Cooling Heat Pump Factor (ehp)',
            'cooling_ave_power_to_HX_W': 'Cooling Average Power to HX (W)',
            'cooling_ave_power_to_HX_MW': 'Cooling Average Power to HX (MW)',
            'cooling_ave_power_to_building_W': 'Cooling Average Power to Building (W)',
            'cooling_monthly_to_HX': 'Cooling Monthly Energy to HX',
            'cooling_monthly_to_building': 'Cooling Monthly Energy to Building',
            'cooling_elec_energy_hydraulic_pumps': 'Cooling Electrical Energy to Hydraulic Pumps',
            'cooling_elec_energy_HP': 'Cooling Electrical Energy to Heat Pump',
            'cooling_annual_elec_energy_J': 'Cooling Annual Electrical Energy (J)',
            'cooling_annual_elec_energy_MWhe': 'Cooling Annual Electrical Energy (MWhe)',
            
            # System parameters
            'energy_balance_ratio': 'Energy Balance Ratio (EBR)',
            'volume_balance_ratio': 'Volume Balance Ratio (VBR)'
        }
        
        return name_mapping.get(param_name, param_name.replace('_', ' ').title())


class ATESResultsExporter:
    """
    Export Monte Carlo results and visualizations
    """
    
    def __init__(self, monte_carlo_results: pd.DataFrame, sensitivity_results: Optional[Dict] = None):
        self.monte_carlo_results = monte_carlo_results
        self.sensitivity_results = sensitivity_results
        self.successful_results = monte_carlo_results[monte_carlo_results['success'] == True] if 'success' in monte_carlo_results.columns else monte_carlo_results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report
        """
        if len(self.successful_results) == 0:
            return {"error": "No successful results to analyze"}
        
        report = {
            "simulation_summary": self._generate_simulation_summary(),
            "statistical_summary": self._generate_statistical_summary(),
            "sensitivity_summary": self._generate_sensitivity_summary() if self.sensitivity_results else None,
        }
        
        return report
    
    
    def _generate_simulation_summary(self) -> Dict:
        """
        Generate simulation summary statistics
        """
        total_iterations = len(self.monte_carlo_results)
        successful_iterations = len(self.successful_results)
        success_rate = successful_iterations / total_iterations * 100 if total_iterations > 0 else 0
        
        return {
            "total_iterations": total_iterations,
            "successful_iterations": successful_iterations,
            "success_rate_percent": round(success_rate, 2),
            "failed_iterations": total_iterations - successful_iterations,
            "convergence_achieved": success_rate > 95
        }
    
    def _generate_statistical_summary(self) -> Dict:
        """
        Generate statistical summary for ALL 62 output parameters with proper infinity handling
        """
        # Get all numeric output parameters 
        numeric_cols = self.successful_results.select_dtypes(include=[np.number]).columns
        all_output_params = [col for col in numeric_cols if col not in ['iteration', 'success']]
        
        summary = {}
        
        for param in all_output_params:
            if param in self.successful_results.columns:
                data = self.successful_results[param].dropna()
                
                if len(data) == 0:
                    continue
                    
                try:
                    # check whether cop related params
                    is_cop_param = ('cop' in param.lower() or 
                                param.endswith('_ehp') or 
                                'heat_pump' in param.lower())
                    
                    if is_cop_param:
                        # seperate give param summary for direct mode or heat pump mode
                        finite_data = data[np.isfinite(data)]
                        infinite_count = len(data) - len(finite_data)
                        
                        param_summary = {
                            "parameter_type": "performance_coefficient",
                            "total_cases": int(len(data))
                        }
                        
                        if infinite_count > 0:
                            param_summary.update({
                                "direct_mode_cases": int(infinite_count),
                                "direct_mode_percentage": round(infinite_count / len(data) * 100, 1),
                                "heat_pump_mode_cases": int(len(finite_data))
                            })
                            
                            if len(finite_data) > 0:
                                param_summary["heat_pump_mode_statistics"] = {
                                    "mean": safe_float(finite_data.mean()),
                                    "std": safe_float(finite_data.std()),
                                    "min": safe_float(finite_data.min()),
                                    "max": safe_float(finite_data.max()),
                                    "percentiles": {
                                        "p0": safe_float(finite_data.quantile(0.00)),
                                        "p10": safe_float(finite_data.quantile(0.10)),
                                        "p20": safe_float(finite_data.quantile(0.20)),
                                        "p30": safe_float(finite_data.quantile(0.30)),
                                        "p40": safe_float(finite_data.quantile(0.40)),
                                        "p50": safe_float(finite_data.quantile(0.50)),
                                        "p60": safe_float(finite_data.quantile(0.60)),
                                        "p70": safe_float(finite_data.quantile(0.70)),
                                        "p80": safe_float(finite_data.quantile(0.80)),
                                        "p90": safe_float(finite_data.quantile(0.90)),
                                        "p100": safe_float(finite_data.quantile(1.00))
                                    },
                                    "distribution_stats": {
                                        "coefficient_of_variation": safe_float(finite_data.std() / finite_data.mean()) if finite_data.mean() != 0 else 0,
                                        "skewness": safe_float(finite_data.skew()) if len(finite_data) > 2 else None
                                    }
                                }
                        else:
                            # no direct mode
                            param_summary.update({
                                "direct_mode_cases": 0,
                                "direct_mode_percentage": 0.0,
                                "heat_pump_mode_cases": int(len(finite_data)),
                                "statistics": {
                                    "mean": safe_float(finite_data.mean()),
                                    "std": safe_float(finite_data.std()),
                                    "min": safe_float(finite_data.min()),
                                    "max": safe_float(finite_data.max()),
                                    "percentiles": {
                                        "p0": safe_float(finite_data.quantile(0.00)),
                                        "p10": safe_float(finite_data.quantile(0.10)),
                                        "p20": safe_float(finite_data.quantile(0.20)),
                                        "p30": safe_float(finite_data.quantile(0.30)),
                                        "p40": safe_float(finite_data.quantile(0.40)),
                                        "p50": safe_float(finite_data.quantile(0.50)),
                                        "p60": safe_float(finite_data.quantile(0.60)),
                                        "p70": safe_float(finite_data.quantile(0.70)),
                                        "p80": safe_float(finite_data.quantile(0.80)),
                                        "p90": safe_float(finite_data.quantile(0.90)),
                                        "p100": safe_float(finite_data.quantile(1.00))
                                    },
                                    "distribution_stats": {
                                        "skewness": safe_float(finite_data.skew()) if len(finite_data) > 2 else None,
                                        "kurtosis": safe_float(finite_data.kurtosis()) if len(finite_data) > 2 else None,
                                        "coefficient_of_variation": safe_float(finite_data.std() / finite_data.mean()) if finite_data.mean() != 0 else 0
                                    }
                                }
                            })
                    else:
                        # non-cop params
                        finite_data = data[np.isfinite(data)]
                        infinite_count = len(data) - len(finite_data)
                        
                        if len(finite_data) > 0:
                            param_summary = {
                                "parameter_type": "standard_output",
                                "count": int(len(finite_data)),
                                "mean": safe_float(finite_data.mean()),
                                "std": safe_float(finite_data.std()),
                                "min": safe_float(finite_data.min()),
                                "max": safe_float(finite_data.max()),
                                "percentiles": {
                                    "p0": safe_float(finite_data.quantile(0.00)),
                                    "p10": safe_float(finite_data.quantile(0.10)),
                                    "p20": safe_float(finite_data.quantile(0.20)),
                                    "p30": safe_float(finite_data.quantile(0.30)),
                                    "p40": safe_float(finite_data.quantile(0.40)),
                                    "p50": safe_float(finite_data.quantile(0.50)),
                                    "p60": safe_float(finite_data.quantile(0.60)),
                                    "p70": safe_float(finite_data.quantile(0.70)),
                                    "p80": safe_float(finite_data.quantile(0.80)),
                                    "p90": safe_float(finite_data.quantile(0.90)),
                                    "p100": safe_float(finite_data.quantile(1.00))
                                },
                                "distribution_stats": {
                                    "skewness": safe_float(finite_data.skew()) if len(finite_data) > 2 else None,
                                    "kurtosis": safe_float(finite_data.kurtosis()) if len(finite_data) > 2 else None,
                                    "coefficient_of_variation": safe_float(finite_data.std() / finite_data.mean()) if finite_data.mean() != 0 else 0
                                }
                            }
                            
                            # record inf without influence stats
                            if infinite_count > 0:
                                param_summary["infinite_values_excluded"] = int(infinite_count)
                                param_summary["note"] = f"{infinite_count} infinite values excluded from statistics"
                    
                    summary[param] = param_summary
                    
                except Exception as e:
                    # if some params return to error, record the others
                    summary[param] = {
                        "parameter_type": "error",
                        "error": f"Failed to calculate statistics: {str(e)}",
                        "raw_count": int(len(data)) if len(data) > 0 else 0
                    }
        
        # stats summary
        total_params = len(summary)
        cop_params = len([p for p, s in summary.items() if s.get("parameter_type") == "performance_coefficient"])
        standard_params = len([p for p, s in summary.items() if s.get("parameter_type") == "standard_output"])
        error_params = len([p for p, s in summary.items() if s.get("parameter_type") == "error"])
        
        summary["_summary_metadata"] = {
            "total_parameters_analyzed": total_params,
            "performance_coefficient_parameters": cop_params,
            "standard_output_parameters": standard_params,
            "parameters_with_errors": error_params
        }
        
        return summary
    

    
    def _generate_sensitivity_summary(self) -> Dict:
        """Generate comprehensive sensitivity analysis summary"""
        if not self.sensitivity_results:
            return {}
        
        # aggregate all correlations
        all_correlations = []
        parameter_influence = {}
        
        for output_param, sensitivity_df in self.sensitivity_results.items():
            for _, row in sensitivity_df.iterrows():
                input_param = row['Input_Parameter']
                abs_corr = safe_float(row['Abs_Pearson'])
                
                all_correlations.append({
                    'output': output_param,
                    'input': input_param,
                    'correlation': abs_corr
                })
                
                if input_param not in parameter_influence:
                    parameter_influence[input_param] = []
                parameter_influence[input_param].append(abs_corr)
        
        # calculate overall parameter importance
        param_importance = {}
        for param, correlations in parameter_influence.items():
            param_importance[param] = {
                "mean_influence": round(float(np.mean(correlations)), 4),
                "max_influence": round(float(np.max(correlations)), 4),
                "min_influence": round(float(np.min(correlations)), 4),
                "std_influence": round(float(np.std(correlations)), 4),
                "outputs_influenced": len(correlations)
            }
        
        # sort by mean influence
        sorted_importance = dict(sorted(param_importance.items(), 
                                        key=lambda x: x[1]['mean_influence'], 
                                        reverse=True))
        
        return {
            "most_influential_parameters": dict(list(sorted_importance.items())[:5]),
            "parameter_influence_summary": {
                "total_parameters_analyzed": len(parameter_influence),
                "average_influence_strength": round(np.mean([np.mean(corrs) for corrs in parameter_influence.values()]), 4),
                "max_influence_strength": round(max([np.max(corrs) for corrs in parameter_influence.values()]), 4)
            },
            "output_sensitivity": {
                output: {
                    "most_sensitive_to": sensitivity_df.iloc[0]['Input_Parameter'],
                    "max_correlation": round(safe_float(sensitivity_df.iloc[0]['Abs_Pearson']), 4),
                    "parameters_analyzed": len(sensitivity_df)
                }
                for output, sensitivity_df in self.sensitivity_results.items()
            }
        }
    
 
# Utility functions for Streamlit integration
def create_results_dashboard():
    """Create a comprehensive results dashboard"""
    if st.session_state.get('monte_carlo_results') is None:
        st.error("No Monte Carlo results available")
        st.info("Please run Monte Carlo analysis in the **Probabilistic Setup** screen first")
        return
    
    visualizer = ATESVisualizer(
        st.session_state.monte_carlo_results, 
        st.session_state.get('sensitivity_results')
    )
    
    # dashboard header
    st.title("ATES Assessment Results Dashboard")
    
    # quick summary metrics
    total_runs = len(st.session_state.monte_carlo_results)
    successful_runs = len(visualizer.successful_results)
    success_rate = (successful_runs / total_runs) * 100 if total_runs > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Iterations", f"{total_runs:,}")
    with col2:
        st.metric("Successful", f"{successful_runs:,}")
    with col3:
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col4:
        has_sensitivity = st.session_state.get('sensitivity_results') is not None
        st.metric("Sensitivity Analysis", "Available" if has_sensitivity else "Not Available")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Distributions", 
        "Percentiles & Risk", 
        "Sensitivity Analysis", 
        "Summary Report"
    ])
    
    with tab1:
        visualizer.render_distribution_plots()
        st.markdown("---")
        visualizer.render_correlation_matrix()
    
    with tab2:
        visualizer.render_percentile_analysis()
    
    with tab3:
        visualizer.render_sensitivity_analysis()
    
    with tab4:
        render_summary_report_tab()


def render_summary_report_tab():
    """
    Render comprehensive summary report tab
    """
    st.subheader("Comprehensive Analysis Report")
    
    if st.session_state.get('monte_carlo_results') is None:
        st.error("No Monte Carlo results available")
        return
    
    # Generate comprehensive report
    exporter = ATESResultsExporter(
        st.session_state.monte_carlo_results, 
        st.session_state.get('sensitivity_results')
    )
    
    report = exporter.generate_comprehensive_report()
    
    if "error" in report:
        st.error(report["error"])
        return
    
    # Executive Summary
    st.markdown("### Executive Summary")
    
    sim_summary = report["simulation_summary"]
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Simulation Success Rate", 
            f"{sim_summary['success_rate_percent']:.1f}%",
            delta="Excellent" if sim_summary['success_rate_percent'] > 95 else "Needs Review"
        )
    
    with col2:
        convergence_status = "Achieved" if sim_summary.get('convergence_achieved', False) else "Check Required"
        st.metric("Convergence", convergence_status)
    
    with col3:
        total_iterations = sim_summary['total_iterations']
        st.metric("Total Simulations", f"{total_iterations:,}")
    
    # Key Performance Indicators
    if "statistical_summary" in report:
        st.markdown("### Key Performance Indicators")
        
        stats_summary = report["statistical_summary"]
        
        
        performance_data = []
        for param, stats in stats_summary.items():
            if param.startswith('_'):  
                continue
                
            param_display = param.replace('_', ' ').title()
            
            try:
               
                if stats.get("parameter_type") == "performance_coefficient":
                
                    if stats.get("heat_pump_mode_statistics"):
                        mean_val = stats["heat_pump_mode_statistics"]["mean"]
                        p10_val = stats["heat_pump_mode_statistics"]["percentiles"]["p10"]
                        p90_val = stats["heat_pump_mode_statistics"]["percentiles"]["p90"]
                    elif stats.get("statistics"):
                        mean_val = stats["statistics"]["mean"]
                        p10_val = stats["statistics"]["percentiles"]["p10"]
                        p90_val = stats["statistics"]["percentiles"]["p90"]
                    else:
                        continue  
                        
                elif stats.get("parameter_type") == "standard_output":
               
                    mean_val = stats["mean"]
                    p10_val = stats["percentiles"]["p10"]
                    p90_val = stats["percentiles"]["p90"]
                else:
                    continue 
                
                performance_data.append({
                    'Parameter': param_display,
                    'Mean': f"{mean_val:.3f}",
                    'P10-P90 Range': f"{p10_val:.3f} - {p90_val:.3f}",
                    'Parameter Type': stats.get("parameter_type", "unknown")
                })
                
            except (KeyError, TypeError):
                continue
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df, width="stretch", hide_index=True)
        else:
            st.info("No valid performance indicators to display")

    
    # Sensitivity Analysis Summary
    if "sensitivity_summary" in report and report["sensitivity_summary"]:
        st.markdown("### Sensitivity Analysis Summary")
        
        sens_summary = report["sensitivity_summary"]
        
        # most influential parameters
        if "most_influential_parameters" in sens_summary:
            st.markdown("**Top 5 Most Influential Parameters:**")
            
            influence_data = []
            for param, influence_stats in sens_summary["most_influential_parameters"].items():
                influence_data.append({
                    'Parameter': param.replace('_', ' ').title(),
                    'Mean Influence': f"{influence_stats['mean_influence']:.4f}",
                    'Max Influence': f"{influence_stats['max_influence']:.4f}",
                    'Outputs Affected': influence_stats['outputs_influenced']
                })
            
            if influence_data:
                influence_df = pd.DataFrame(influence_data)
                st.dataframe(influence_df, width="stretch", hide_index=True)
        
        # sensitivity statistics
        if "parameter_influence_summary" in sens_summary:
            influence_summary = sens_summary["parameter_influence_summary"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Parameters Analyzed", influence_summary['total_parameters_analyzed'])
            with col2:
                st.metric("Average Influence", f"{influence_summary['average_influence_strength']:.4f}")
            with col3:
                st.metric("Max Influence", f"{influence_summary['max_influence_strength']:.4f}")
    

    
    # Export Options
    st.markdown("### Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # export report as JSON
        report_json = json.dumps(report, indent=2, default=str)
        st.download_button(
            label="Download Full Report (JSON)",
            data=report_json,
            file_name=f"ates_comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="download_json_report",
            width="stretch"
        )
    
    with col2:
        # export Monte Carlo results as CSV
        csv_data = st.session_state.monte_carlo_results.to_csv(index=False)
        st.download_button(
            label="Download Results Data (CSV)",
            data=csv_data,
            file_name=f"ates_monte_carlo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_csv_data",
            width="stretch"
        )
    
    with col3:
        # export sensitivity analysis if available
        if st.session_state.get('sensitivity_results'):
            # combine all sensitivity results into one CSV
            all_sensitivity = []
            for output_param, sens_df in st.session_state.sensitivity_results.items():
                sens_df_copy = sens_df.copy()
                sens_df_copy['Output_Parameter'] = output_param
                all_sensitivity.append(sens_df_copy)
            
            combined_sensitivity = pd.concat(all_sensitivity, ignore_index=True)
            sensitivity_csv = combined_sensitivity.to_csv(index=False)
            
            st.download_button(
                label="Download Sensitivity Data (CSV)",
                data=sensitivity_csv,
                file_name=f"ates_sensitivity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_sensitivity_csv",
                width="stretch"
            )
        else:
            st.button(
                "Download Sensitivity Data (CSV)",
                disabled=True,
                help="No sensitivity data available",
                width="stretch"
            )


# Additional utility functions
def format_monte_carlo_summary(results_df: pd.DataFrame) -> str:
    """
    Format a summary of Monte Carlo results for display
    """
    if results_df is None or len(results_df) == 0:
        return "No results available"
    
    successful = int(results_df['success'].sum()) if 'success' in results_df.columns else len(results_df)
    total = len(results_df)
    success_rate = successful / total * 100
    
    summary = f"""**Monte Carlo Simulation Summary**
- Total iterations: {total:,}
- Successful calculations: {successful:,} ({success_rate:.1f}%)
- Failed calculations: {total - successful:,}"""
    
    if successful > 0:
        successful_df = results_df[results_df['success']] if 'success' in results_df.columns else results_df
        
        # Safe access to columns
        heating_cop_mean = safe_float(successful_df['heating_system_cop'].mean()) if 'heating_system_cop' in successful_df.columns else 0
        cooling_cop_mean = safe_float(successful_df['cooling_system_cop'].mean()) if 'cooling_system_cop' in successful_df.columns else 0
        
        # Handle infinite values (direct cooling mode)
        if 'cooling_system_cop' in successful_df.columns:
            cooling_finite = successful_df['cooling_system_cop'][successful_df['cooling_system_cop'] != float('inf')]
            if len(cooling_finite) > 0:
                cooling_cop_display = f"{safe_float(cooling_finite.mean()):.2f} (finite values)"
            else:
                cooling_cop_display = "Direct cooling mode"
        else:
            cooling_cop_display = "N/A"
        
        summary += f"""

**Key Results (Mean Values)**
- Heating System COP: {heating_cop_mean:.2f}
- Cooling System COP: {cooling_cop_display}
"""
    
    return summary


def create_quick_visualization(param_name: str, data: pd.Series, title: Optional[str] = None) -> go.Figure:
    """
    create a quick visualization for a single parameter
    """
    if title is None:
        title = f"Distribution: {param_name.replace('_', ' ').title()}"
    
    # create subplot with histogram and box plot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Distribution", "Statistics"],
        column_widths=[0.7, 0.3]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=data, nbinsx=30, name="Frequency", showlegend=False),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(y=data, name="Statistics", showlegend=False),
        row=1, col=2
    )
    
    # Add mean line with safe conversion
    mean_val = safe_float(data.mean())
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {mean_val:.3f}")
    
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        height=400
    )
    
    return fig


# export the main classes and functions
__all__ = [
    'ATESVisualizer',
    'ATESResultsExporter', 
    'create_results_dashboard',
    'render_summary_report_tab',
    'format_monte_carlo_summary',
    'create_quick_visualization',
    'safe_float',
    'safe_int',
    'safe_abs'
]