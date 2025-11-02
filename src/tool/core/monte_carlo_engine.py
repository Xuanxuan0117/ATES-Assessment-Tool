"""
ATES Monte Carlo Engine
"""
# import libraries
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
import scipy.stats as stats
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings

# import calculator
from tool.core.ates_calculator import ATESParameters, ATESCalculator, ATESResults


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation"""
    iterations: int = 10000
    seed: Optional[int] = None
    parallel: bool = True
    max_workers: int = 4
    chunk_size: int = 1000

@dataclass
class DistributionParams:
    """Parameters for probability distributions"""
    type: str
    value: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    most_likely: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    location: float = 0.0  # Added for lognormal distribution
    use_log_params: bool = False  # Flag to indicate if mean/std are already in log space

class ParameterSampler:
    """
    Handles sampling from different probability distributions
    """
    
    @staticmethod
    def sample_parameter(dist_params: DistributionParams, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Sample from a parameter distribution"""
        # Get the distribution type
        dist_type = dist_params.type

        # 1. Single fixed value, fill an array with the same constant value
        if dist_type == 'single_value':
            return np.full(n_samples, dist_params.value, dtype=np.float64)
        
        # 2. Uniform distribution, sample uniformly from min value to max value
        elif dist_type == 'range':
            samples = rng.uniform(dist_params.min_val, dist_params.max_val, n_samples)
            return np.asarray(samples, dtype=np.float64)
        
        # 3. Triangular distribution 
        elif dist_type == 'triangular':
            # If min and max are almost equal, treat it as a single value
            if abs(dist_params.max_val - dist_params.min_val) < 1e-10:
                return np.full(n_samples, dist_params.min_val, dtype=np.float64)
            # Compute the relative peak position c
            c = (dist_params.most_likely - dist_params.min_val) / (dist_params.max_val - dist_params.min_val)
            c = np.clip(c, 0.0, 1.0)
            # Sample from a triangular distribution using scipy
            samples = stats.triang.rvs(
                c, loc=dist_params.min_val, 
                scale=dist_params.max_val - dist_params.min_val, 
                size=n_samples, random_state=rng
            )
            return np.asarray(samples, dtype=np.float64)
        
        # 4. Gaussian distribution
        elif dist_type == 'normal':
            samples = rng.normal(dist_params.mean, dist_params.std, n_samples)
            return np.asarray(samples, dtype=np.float64)
        
        # 5. Lognormal distribution
        elif dist_type == 'lognormal':
        # Extract parameters
            location = float(dist_params.location)  # Crystal Ball's location parameter
            m = float(dist_params.mean)
            s = float(dist_params.std)
            use_log_params = dist_params.use_log_params
            
            # Validate inputs
            if m <= 0 or s <= 0:
                raise ValueError("Mean and std must be positive for lognormal distribution")
            
            if use_log_params:
                # Parameters are already log mean and log std (Crystal Ball's recommended approach for historical data)
                mu = m
                sigma = s
                warnings.warn("Using log-space parameters as recommended by Crystal Ball for historical data")
            else:
                # Convert arithmetic mean/std to log parameters (Crystal Ball's default mode)
                if m <= location:
                    raise ValueError("Mean must be greater than location parameter for lognormal distribution")
                
                # Adjust mean by location parameter
                adjusted_mean = m - location
                
                # Convert to log-space parameters using moment matching
                sigma = np.sqrt(np.log(1.0 + (s * s) / (adjusted_mean * adjusted_mean)))
                mu = np.log(adjusted_mean) - 0.5 * sigma * sigma
            
            # Generate samples and add location shift
            samples = rng.lognormal(mean=mu, sigma=sigma, size=n_samples) + location
            return np.asarray(samples, dtype=np.float64)

        
        # 6. Unknown distribution type raise error
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
        
class ATESMonteCarloEngine:
    """
    Main Monte Carlo simulation engine for ATES calculations
    """
    
    def __init__(self, base_parameters: ATESParameters, config: Optional[MonteCarloConfig] = None):
        # disable physical constrains for monte carlo simulation
        ATESParameters.disable_validation()
        # store the baseline deterministic parameters for all simulations
        self.base_parameters = base_parameters
        # use provided config or use default MonteCarloConfig if none given.
        self.config = config if config is not None else MonteCarloConfig()
        # main simulation output df
        self.results: Optional[pd.DataFrame] = None
        # sensitivity analysis outputs df
        self.sensitivity_results: Optional[Dict[str, pd.DataFrame]] = None

    def _generate_parameter_samples(self, parameter_distributions: Dict[str, Dict[str, Any]], 
                                rng: np.random.Generator) -> pd.DataFrame:
        samples = {}
        
        for param_name, dist_config in parameter_distributions.items():
            dist_type = dist_config['type']
            
            if dist_type == 'single_value':
                
                samples[param_name] = np.full(
                    self.config.iterations, 
                    dist_config['value'], 
                    dtype=np.float64
                )
            else:
                
                dist_params = DistributionParams(
                    type=dist_type,
                    value=float(dist_config.get('value', 0.0)),
                    min_val=float(dist_config.get('min', 0.0)),
                    max_val=float(dist_config.get('max', 0.0)),
                    most_likely=float(dist_config.get('most_likely', 0.0)),
                    mean=float(dist_config.get('mean', 0.0)),
                    std=float(dist_config.get('std', 0.0)),
                    location=float(dist_config.get('location', 0.0)),
                    use_log_params=bool(dist_config.get('use_log_params', False))
                )
                samples[param_name] = ParameterSampler.sample_parameter(
                    dist_params, self.config.iterations, rng
                )
        
        return pd.DataFrame(samples)
    
    def _create_parameter_instance(self, parameter_row: pd.Series) -> ATESParameters:
        """
        Create ATESParameters instance with sampled values
        """
        # build a fresh parameters from base
        params = ATESParameters(
            aquifer_temp=self.base_parameters.aquifer_temp,
            water_density=self.base_parameters.water_density,
            water_specific_heat_capacity=self.base_parameters.water_specific_heat_capacity,
            thermal_recovery_factor=self.base_parameters.thermal_recovery_factor,
            heating_target_avg_flowrate_pd=self.base_parameters.heating_target_avg_flowrate_pd,
            tolerance_in_energy_balance=self.base_parameters.tolerance_in_energy_balance,
            heating_number_of_doublets=self.base_parameters.heating_number_of_doublets,
            heating_months=self.base_parameters.heating_months,
            cooling_months=self.base_parameters.cooling_months,
            pump_energy_density=self.base_parameters.pump_energy_density,
            heating_ave_injection_temp=self.base_parameters.heating_ave_injection_temp,
            heating_temp_to_building=self.base_parameters.heating_temp_to_building,
            cop_param_a=self.base_parameters.cop_param_a,
            cop_param_b=self.base_parameters.cop_param_b,
            cop_param_c=self.base_parameters.cop_param_c,
            cop_param_d=self.base_parameters.cop_param_d,
            carbon_intensity=self.base_parameters.carbon_intensity,
            cooling_ave_injection_temp=self.base_parameters.cooling_ave_injection_temp,
            cooling_temp_to_building=self.base_parameters.cooling_temp_to_building
        )
        # override the copied parameter with the iteration's sampled values where applicable
        for param_name, value in parameter_row.items():
            param_str = str(param_name)
            if hasattr(params, param_str):
                # set the sampled valye
                setattr(params, param_str, float(value))
        # recompute derived fields
        params.__post_init__()
        return params
    
    ## replace method
    # def _create_parameter_instance(self, parameter_row: pd.Series) -> ATESParameters:
        """Create ATESParameters instance by cloning the base and applying sampled overrides."""
        # shallow copy the base to preserve all existing baseline values (including any non-default overrides)
        params = replace(self.base_parameters)

        # override only the sampled uncertain inputs from this Monte Carlo draw
        for param_name, value in parameter_row.items():
            param_str = str(param_name)
            if hasattr(params, param_str):
                try:
                    setattr(params, param_str, float(value))
                except Exception:
                    warnings.warn(f"Failed to set parameter '{param_str}' to {value}; keeping previous value.")
            else:
                # 
                warnings.warn(f"Sampled parameter '{param_str}' does not match any ATESParameters attribute.")

        # Recompute all derived fields and run validation based on the updated base inputs
        params.__post_init__()
        return params

    def _extract_results(self, result: ATESResults, iteration: int) -> Dict[str, Any]:
        """
        extract ALL 62 results from ATESResults object with enhanced anomaly classification
        """
        
        # classify abnormal situations
        heating_anomaly = self._classify_heating_anomaly(result)
        cooling_anomaly = self._classify_cooling_anomaly(result)
        
        return {
            'iteration': iteration,
            'success': True,
            
            # label abnormal situation
            'heating_anomaly_type': heating_anomaly,
            'cooling_anomaly_type': cooling_anomaly,
            'has_parameter_error': heating_anomaly == 'parameter_error' or cooling_anomaly == 'parameter_error',
            'has_cop_failure': heating_anomaly == 'cop_failure' or cooling_anomaly == 'cop_failure',
            'has_direct_mode': heating_anomaly == 'direct_mode' or cooling_anomaly == 'direct_mode',
            
            # HEATING OUTPUTS (K Column) - 32 parameters
            'heating_total_energy_stored': float(result.heating_total_energy_stored),                           # K3
            'heating_stored_energy_recovered': float(result.heating_stored_energy_recovered),                     # K4
            'heating_total_flow_rate_m3hr': float(result.heating_total_flow_rate_m3hr),                          # K6
            'heating_total_flow_rate_ls': float(result.heating_total_flow_rate_ls),                              # K7
            'heating_total_flow_rate_m3s': float(result.heating_total_flow_rate_m3s),                            # K8
            'heating_ave_production_temp': float(result.heating_ave_production_temp),                            # K10
            'heating_ave_temp_change_across_HX': float(result.heating_ave_temp_change_across_HX),                # K11
            'heating_temp_change_induced_HP': float(result.heating_temp_change_induced_HP),                      # K12
            'heating_heat_pump_COP': float(result.heating_heat_pump_COP),                                        # K13
            'heating_ehp': float(result.heating_ehp),                                                            # K14
            'heating_ave_power_to_HX_W': float(result.heating_ave_power_to_HX_W),                                # K16
            'heating_ave_power_to_HX_MW': float(result.heating_ave_power_to_HX_MW),                              # K17
            'heating_annual_energy_aquifer_J': float(result.heating_annual_energy_aquifer_J),                    # K19
            'heating_annual_energy_aquifer_kWhth': float(result.heating_annual_energy_aquifer_kWhth),            # K20
            'heating_annual_energy_aquifer_GWhth': float(result.heating_annual_energy_aquifer_GWhth),            # K21
            'heating_monthly_to_HX': float(result.heating_monthly_to_HX),                                        # K22
            'energy_balance_ratio': float(result.energy_balance_ratio),                                          # K23
            'volume_balance_ratio': float(result.volume_balance_ratio),                                          # K24
            'heating_ave_power_to_building_W': float(result.heating_ave_power_to_building_W),                    # K26
            'heating_ave_power_to_building_MW': float(result.heating_ave_power_to_building_MW),                  # K27
            'heating_annual_energy_building_J': float(result.heating_annual_energy_building_J),                  # K29
            'heating_annual_energy_building_kWhth': float(result.heating_annual_energy_building_kWhth),          # K30
            'heating_annual_energy_building_GWhth': float(result.heating_annual_energy_building_GWhth),          # K31
            'heating_monthly_to_building': float(result.heating_monthly_to_building),                            # K32
            'heating_elec_energy_hydraulic_pumps': float(result.heating_elec_energy_hydraulic_pumps),            # K34
            'heating_elec_energy_HP': float(result.heating_elec_energy_HP),                                      # K35
            'heating_annual_elec_energy_J': float(result.heating_annual_elec_energy_J),                          # K36
            'heating_annual_elec_energy_MWhe': float(result.heating_annual_elec_energy_MWhe),                    # K37
            'heating_annual_elec_energy_GWhe': float(result.heating_annual_elec_energy_GWhe),                    # K38
            'heating_system_cop': float(result.heating_system_cop),                                              # K39
            'heating_elec_energy_per_thermal': float(result.heating_elec_energy_per_thermal),                    # K40
            'heating_co2_emissions_per_thermal': float(result.heating_co2_emissions_per_thermal),                # K41
            
            # COOLING OUTPUTS (N Column) - 30 parameters + 1
            'cooling_total_energy_stored': float(result.cooling_total_energy_stored),                            # N3
            'cooling_stored_energy_recovered': float(result.cooling_stored_energy_recovered),                    # N4
            'cooling_target_avg_flowrate_pd': float(result.cooling_target_avg_flowrate_pd),                      # G10 # type: ignore
            'cooling_total_flow_rate_m3hr': float(result.cooling_total_flow_rate_m3hr),                          # N6
            'cooling_total_flow_rate_ls': float(result.cooling_total_flow_rate_ls),                              # N7
            'cooling_total_flow_rate_m3s': float(result.cooling_total_flow_rate_m3s),                            # N8
            'cooling_ave_production_temp': float(result.cooling_ave_production_temp),                            # N10
            'cooling_ave_temp_change_across_HX': float(result.cooling_ave_temp_change_across_HX),                # N11
            'cooling_temp_change_induced_HP': float(result.cooling_temp_change_induced_HP),                      # N12
            'cooling_heat_pump_COP': float(result.cooling_heat_pump_COP),                                        # N13
            'cooling_ehp': float(result.cooling_ehp),                                                            # N14
            'cooling_ave_power_to_HX_W': float(result.cooling_ave_power_to_HX_W),                                # N16
            'cooling_ave_power_to_HX_MW': float(result.cooling_ave_power_to_HX_MW),                              # N17
            'cooling_annual_energy_aquifer_J': float(result.cooling_annual_energy_aquifer_J),                    # N19
            'cooling_annual_energy_aquifer_kWhth': float(result.cooling_annual_energy_aquifer_kWhth),            # N20
            'cooling_annual_energy_aquifer_GWhth': float(result.cooling_annual_energy_aquifer_GWhth),            # N21
            'cooling_monthly_to_HX': float(result.cooling_monthly_to_HX),                                        # N22
            'cooling_ave_power_to_building_W': float(result.cooling_ave_power_to_building_W),                    # N26
            'cooling_ave_power_to_building_MW': float(result.cooling_ave_power_to_building_MW),                  # N27
            'cooling_annual_energy_building_J': float(result.cooling_annual_energy_building_J),                  # N29
            'cooling_annual_energy_building_kWhth': float(result.cooling_annual_energy_building_kWhth),          # N30
            'cooling_annual_energy_building_GWhth': float(result.cooling_annual_energy_building_GWhth),          # N31
            'cooling_monthly_to_building': float(result.cooling_monthly_to_building),                            # N32
            'cooling_elec_energy_hydraulic_pumps': float(result.cooling_elec_energy_hydraulic_pumps),            # N34
            'cooling_elec_energy_HP': float(result.cooling_elec_energy_HP),                                      # N35
            'cooling_annual_elec_energy_J': float(result.cooling_annual_elec_energy_J),                          # N36
            'cooling_annual_elec_energy_MWhe': float(result.cooling_annual_elec_energy_MWhe),                    # N37
            'cooling_annual_elec_energy_GWhe': float(result.cooling_annual_elec_energy_GWhe),                    # N38
            'cooling_system_cop': float(result.cooling_system_cop),                                              # N39
            'cooling_elec_energy_per_thermal': float(result.cooling_elec_energy_per_thermal),                    # N40
            'cooling_co2_emissions_per_thermal': float(result.cooling_co2_emissions_per_thermal),                # N41
            
            # derived output
            'heating_direct_mode': bool(getattr(result, 'heating_direct_mode', False)),
            'cooling_direct_mode': bool(getattr(result, 'cooling_direct_mode', False))
        }

    def _classify_heating_anomaly(self, result: ATESResults) -> str:
        """classify heating anomaly"""
        try:
            # Situation1: Normal Direct Mode
            if getattr(result, 'heating_direct_mode', False):
                return 'direct_mode'
            
            # Situation 3: COP failure(ehp = inf and System COP = NaN)
            if (hasattr(result, 'heating_ehp') and 
                result.heating_ehp == float('inf') and
                np.isnan(result.heating_system_cop)):
                return 'cop_failure'
            
            # Situation 2: parameter error (negative electrical energy or System COP = inf but not direct mode)
            if (result.heating_system_cop == float('inf') or 
                (hasattr(result, 'heating_annual_elec_energy_GWhe') and result.heating_annual_elec_energy_GWhe < 0) or
                (hasattr(result, 'heating_elec_energy_hydraulic_pumps') and result.heating_elec_energy_hydraulic_pumps < 0)):
                return 'parameter_error'
            
            # normal situation
            if (hasattr(result, 'heating_system_cop') and 
                np.isfinite(result.heating_system_cop) and 
                result.heating_system_cop > 0):
                return 'normal'
            
            # other situation
            return 'unknown'
        
        except Exception as e:
            return 'classification_error'

    def _classify_cooling_anomaly(self, result: ATESResults) -> str:
        """classify cooling anomaly"""
        try:
            # Situation1: Normal Direct Mode
            if getattr(result, 'cooling_direct_mode', False):
                return 'direct_mode'
            
            # Situation 3: COP failure(ehp = inf and System COP = NaN)
            if (hasattr(result, 'cooling_ehp') and 
                result.cooling_ehp == float('inf') and
                np.isnan(result.cooling_system_cop)):
                return 'cop_failure'
            
            # Situation 2: parameter error (negative electrical energy or System COP = inf but not direct mode)
            if (result.cooling_system_cop == float('inf') or 
                (hasattr(result, 'cooling_annual_elec_energy_GWhe') and result.cooling_annual_elec_energy_GWhe < 0) or
                (hasattr(result, 'cooling_elec_energy_hydraulic_pumps') and result.cooling_elec_energy_hydraulic_pumps < 0)):
                return 'parameter_error'
            
            # normal situation
            if (hasattr(result, 'cooling_system_cop') and 
                np.isfinite(result.cooling_system_cop) and 
                result.cooling_system_cop > 0):
                return 'normal'
            
            # other situation
            return 'unknown'
        
        except Exception as e:
            return 'classification_error'


    def _create_error_result(self, iteration: int, error_msg: str) -> Dict[str, Any]:
        """
        Create result dictionary for failed calculations
        """
        return {
            'iteration': iteration,
            'success': False,
            'error': error_msg,
            
            # calculation error
            'heating_anomaly_type': 'calculation_error',
            'cooling_anomaly_type': 'calculation_error',
            'has_parameter_error': False,  
            'has_cop_failure': False,
            'has_direct_mode': False,
        
            'heating_system_cop': np.nan,
            'heating_annual_energy_building_GWhth': np.nan,
            'heating_annual_elec_energy_GWhe': np.nan,
            'heating_co2_emissions_per_thermal': np.nan,
            'heating_ave_power_to_building_MW': np.nan,
            'heating_ave_production_temp': np.nan,
            'heating_direct_mode': False,
            
            'cooling_system_cop': np.nan,
            'cooling_target_avg_flowrate_pd': np.nan,
            'cooling_annual_energy_building_GWhth': np.nan,
            'cooling_annual_elec_energy_GWhe': np.nan,
            'cooling_co2_emissions_per_thermal': np.nan,
            'cooling_ave_power_to_building_MW': np.nan,
            'cooling_ave_production_temp': np.nan,
            'cooling_direct_mode': False,
            
            'energy_balance_ratio': np.nan,
            'volume_balance_ratio': np.nan,
            

            'heating_total_energy_stored': np.nan,
            'heating_stored_energy_recovered': np.nan,
            'heating_total_flow_rate_m3hr': np.nan,
            'heating_total_flow_rate_ls': np.nan,
            'heating_total_flow_rate_m3s': np.nan,
            'heating_ave_temp_change_across_HX': np.nan,
            'heating_temp_change_induced_HP': np.nan,
            'heating_heat_pump_COP': np.nan,
            'heating_ehp': np.nan,
            'heating_ave_power_to_HX_W': np.nan,
            'heating_ave_power_to_HX_MW': np.nan,
            'heating_annual_energy_aquifer_J': np.nan,
            'heating_annual_energy_aquifer_kWhth': np.nan,
            'heating_annual_energy_aquifer_GWhth': np.nan,
            'heating_monthly_to_HX': np.nan,
            'heating_ave_power_to_building_W': np.nan,
            'heating_annual_energy_building_J': np.nan,
            'heating_annual_energy_building_kWhth': np.nan,
            'heating_monthly_to_building': np.nan,
            'heating_elec_energy_hydraulic_pumps': np.nan,
            'heating_elec_energy_HP': np.nan,
            'heating_annual_elec_energy_J': np.nan,
            'heating_annual_elec_energy_MWhe': np.nan,
            'heating_elec_energy_per_thermal': np.nan,
            
            'cooling_total_energy_stored': np.nan,
            'cooling_stored_energy_recovered': np.nan,
            'cooling_total_flow_rate_m3hr': np.nan,
            'cooling_total_flow_rate_ls': np.nan,
            'cooling_total_flow_rate_m3s': np.nan,
            'cooling_ave_temp_change_across_HX': np.nan,
            'cooling_temp_change_induced_HP': np.nan,
            'cooling_heat_pump_COP': np.nan,
            'cooling_ehp': np.nan,
            'cooling_ave_power_to_HX_W': np.nan,
            'cooling_ave_power_to_HX_MW': np.nan,
            'cooling_annual_energy_aquifer_J': np.nan,
            'cooling_annual_energy_aquifer_kWhth': np.nan,
            'cooling_annual_energy_aquifer_GWhth': np.nan,
            'cooling_monthly_to_HX': np.nan,
            'cooling_ave_power_to_building_W': np.nan,
            'cooling_ave_power_to_building_MW': np.nan,
            'cooling_annual_energy_building_J': np.nan,
            'cooling_annual_energy_building_kWhth': np.nan,
            'cooling_monthly_to_building': np.nan,
            'cooling_elec_energy_hydraulic_pumps': np.nan,
            'cooling_elec_energy_HP': np.nan,
            'cooling_annual_elec_energy_J': np.nan,
            'cooling_annual_elec_energy_MWhe': np.nan,
            'cooling_elec_energy_per_thermal': np.nan
        }


    def _run_sequential_calculations(self, parameter_samples: pd.DataFrame, 
                                   progress_callback: Optional[Callable[[int, int], None]] = None) -> pd.DataFrame:
        """
        Run calculations sequentially
        """
        # initialize a list to storage the output dict of every sample
        results: List[Dict[str, Any]] = []
        
        for i, (_, row) in enumerate(parameter_samples.iterrows()):
            params = self._create_parameter_instance(row)
            # if success, put into calculator; if failure, put into error result
            try:
                calculator = ATESCalculator(params)
                result = calculator.calculate()
                result_dict = self._extract_results(result, i)
                results.append(result_dict)
            except Exception as e:
                result_dict = self._create_error_result(i, str(e))
                results.append(result_dict)
            # recording the progress
            if progress_callback:
                progress_callback(i + 1, self.config.iterations)
        # put all rounds results in to a df
        return pd.DataFrame(results)
    
    def _process_chunk(self, chunk: pd.DataFrame, start_index: int) -> List[Dict[str, Any]]:
        """
        Process a chunk of parameter samples
        """
        # accumulate results for this chunk
        chunk_results: List[Dict[str, Any]] = []
        
        for i, (_, row) in enumerate(chunk.iterrows()):
            # build a full instance for this sample
            params = self._create_parameter_instance(row)
            
            try:
                calculator = ATESCalculator(params)
                result = calculator.calculate()
                # extract and flatten outputs, using global iteration index
                result_dict = self._extract_results(result, start_index + i)
                chunk_results.append(result_dict)
            except Exception as e:
                result_dict = self._create_error_result(start_index + i, str(e))
                chunk_results.append(result_dict)
        # return all results from this chunk to the caller for aggregation
        return chunk_results


    def _run_parallel_calculations(self, parameter_samples: pd.DataFrame,
                             progress_callback: Optional[Callable[[int, int], None]] = None) -> pd.DataFrame:
        """Run calculations in parallel with guaranteed iteration order"""
        results: List[Dict[str, Any]] = []
        completed = 0
        
        # split samples into chunks of configured size
        chunks = [
            parameter_samples.iloc[i:i + self.config.chunk_size]
            for i in range(0, len(parameter_samples), self.config.chunk_size)
        ]
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # correctly associate each chunk with its global start index
            future_to_info = {
                executor.submit(self._process_chunk, chunk, idx * self.config.chunk_size): (chunk, idx * self.config.chunk_size)
                for idx, chunk in enumerate(chunks)
            }

            for future in as_completed(future_to_info):
                chunk, start_index = future_to_info[future]
                try:
                    chunk_results = future.result()  # list of dicts for this chunk
                    results.extend(chunk_results)     # append to master list
                    completed += len(chunk_results)   # increment count
                    if progress_callback:
                        progress_callback(completed, self.config.iterations)
                except Exception as e:
                    # Entire chunk failed unexpectedly; fallback to per-iteration error entries
                    err_msg = f"Chunk starting at {start_index} crashed: {e}"
                    chunk_size = len(chunk)
                    for i in range(chunk_size):
                        results.append(self._create_error_result(start_index + i, err_msg))
        
        # create df ang sort by the iteration order
        results_df = pd.DataFrame(results)
        
        if 'iteration' in results_df.columns and len(results_df) > 0:
            results_df = results_df.sort_values('iteration', ascending=True).reset_index(drop=True)
            
           
            expected_iterations = list(range(self.config.iterations))
            actual_iterations = sorted(results_df['iteration'].unique())
            
            if actual_iterations != expected_iterations:
                print(f"Warning: Iteration sequence may have gaps. Expected {len(expected_iterations)}, got {len(actual_iterations)}")
        
        return results_df
            

    def run_simulation(self, parameter_distributions: Dict[str, Dict[str, Any]], 
                    progress_callback: Optional[Callable[[int, int], None]] = None) -> pd.DataFrame:
        
        ATESParameters.disable_validation()
        
        try:
            if self.config.seed is not None:
                np.random.seed(self.config.seed)
            rng = np.random.default_rng(self.config.seed)
            
            parameter_samples = self._generate_parameter_samples(parameter_distributions, rng)
            
            if self.config.parallel and self.config.iterations > 1000:
                results = self._run_parallel_calculations(parameter_samples, progress_callback)
            else:
                results = self._run_sequential_calculations(parameter_samples, progress_callback)
            
            self.results = results
            return results
            
        finally:
            ATESParameters.enable_validation()
    
    def calculate_statistics(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate summary statistics for all output parameters
        """
        # set guard
        if self.results is None:
            raise ValueError("No results available. Run simulation first.")
        # only save success iteration
        successful_results = self.results[self.results['success'] == True].copy()
        # if all iterations are fail, raise error in advance to avoid misleading output
        if len(successful_results) == 0:
            raise ValueError("No successful calculations in Monte Carlo results")
        # simple group by the name start with heating/cooling except ebr, vbr, etc
        parameter_groups = {
            'heating': [col for col in successful_results.columns if col.startswith('heating_')],
            'cooling': [col for col in successful_results.columns if col.startswith('cooling_')],
            'system': ['energy_balance_ratio', 'volume_balance_ratio', 'total_annual_energy_gwh', 
                      'total_electrical_energy_gwh', 'overall_system_cop']
        }
        # create a dict to put all statistics
        statistics: Dict[str, pd.DataFrame] = {}
        # one single table for each group
        for group_name, params in parameter_groups.items():
            group_data = successful_results[params]
            stats_data: List[Dict[str, Any]] = []
            # iterate every group and prepare to collect the summary of each parameter in this gp
            for param in params:
                if param in group_data.columns:
                    param_data = group_data[param].dropna() # drop missing value
                    if len(param_data) > 0:
                        stats_data.append({
                            'Parameter': param,
                            'Mean': float(param_data.mean()),
                            'Std': float(param_data.std()),
                            'Min': float(param_data.min()),
                            'P0': float(param_data.quantile(0.00)),   # 0% 
                            'P10': float(param_data.quantile(0.10)),  # 10%
                            'P20': float(param_data.quantile(0.20)),  # 20%
                            'P30': float(param_data.quantile(0.30)),  # 30%
                            'P40': float(param_data.quantile(0.40)),  # 40%
                            'P50': float(param_data.quantile(0.50)),  # 50% 
                            'P60': float(param_data.quantile(0.60)),  # 60%
                            'P70': float(param_data.quantile(0.70)),  # 70%
                            'P80': float(param_data.quantile(0.80)),  # 80%
                            'P90': float(param_data.quantile(0.90)),  # 90%
                            'P100': float(param_data.quantile(1.00)), # 100% 
                            'Max': float(param_data.max()),
                            'Count': int(len(param_data))
                        })
            
            statistics[group_name] = pd.DataFrame(stats_data)
        
        return statistics
    
    def calculate_sensitivity_analysis(self, parameter_samples: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate sensitivity analysis using correlation coefficients, properly handling anomalies.
        Only analyzes normal cases and direct mode cases, excluding parameter errors and COP failures.
        """
        # validate that Monte Carlo results are available
        if self.results is None:
            raise ValueError("No results available. Run simulation first.")
        
        # filter to only successful calculation iterations
        successful_results = self.results[self.results['success'] == True].copy()
        
        if len(successful_results) == 0:
            raise ValueError("No successful calculations for sensitivity analysis")
        
        # Get mask for analyzable cases (normal + direct mode, exclude errors)
        analyzable_mask = self._get_analyzable_cases_mask(successful_results)
        analyzable_results = successful_results[analyzable_mask].copy()
        
        if len(analyzable_results) < 10:
            warnings.warn(f"Only {len(analyzable_results)} analyzable cases for sensitivity analysis. Results may be unreliable.")
        
        # ensure alignment between parameter samples and analyzable results
        successful_indices = analyzable_results['iteration'].values
        max_index = len(parameter_samples) - 1
        
        # convert to numpy array and validate indices are within parameter sample range
        indices_array = np.asarray(successful_indices, dtype=int)
        valid_mask = indices_array <= max_index
        valid_indices = indices_array[valid_mask]
        
        if len(valid_indices) == 0:
            raise ValueError("No valid indices for sensitivity analysis")
        
        # align parameter samples with analyzable results based on iteration indices
        param_samples_analyzable = parameter_samples.iloc[valid_indices].copy()
        analyzable_results_filtered = analyzable_results.iloc[:len(valid_indices)].copy()
        
        # automatically detect all numerical output parameters, excluding classification columns
        numeric_columns = analyzable_results_filtered.select_dtypes(include=[np.number]).columns
        output_params = [col for col in numeric_columns 
                        if col not in ['iteration', 'success'] 
                        and not col.endswith('_anomaly_type')
                        and not col.startswith('has_')]
        
        print(f"Sensitivity analysis: Processing {len(output_params)} output parameters on {len(analyzable_results_filtered)} analyzable cases")
        
        # Log case breakdown
        self._log_case_breakdown(successful_results)
        
        # initialize results dictionary to store correlation analysis for each output parameter
        sensitivity_results: Dict[str, pd.DataFrame] = {}
        
        # perform correlation analysis for each output parameter
        for output_param in output_params:
            output_values = analyzable_results_filtered[output_param].values
            correlations: List[Dict[str, Any]] = []
            
            # Special handling for parameters that might have infinite values in direct mode
            output_values_processed = self._preprocess_output_values(output_param, output_values, analyzable_results_filtered)
            
            # calculate correlations between this output and each input parameter
            for input_param in param_samples_analyzable.columns:
                input_values = param_samples_analyzable[input_param].values
                
                # ensure data arrays have the same length for correlation calculation
                min_length = min(len(input_values), len(output_values_processed))
                if min_length < 3:  # Need minimum 3 points for meaningful correlation
                    continue
                
                try:
                    # convert to numpy arrays with consistent data type
                    input_data = np.array(input_values[:min_length], dtype=np.float64)
                    output_data = np.array(output_values_processed[:min_length], dtype=np.float64)
                    
                    # remove invalid values including NaN and infinite values
                    valid_mask = np.isfinite(input_data) & np.isfinite(output_data)
                    if np.sum(valid_mask) < 3:
                        continue
                    
                    input_clean = input_data[valid_mask]
                    output_clean = output_data[valid_mask]
                    
                    # calculate Pearson correlation coefficient
                    pearson_corr = 0.0
                    try:
                        if len(input_clean) > 1 and np.std(input_clean) > 0 and np.std(output_clean) > 0:
                            corr_matrix = np.corrcoef(input_clean, output_clean)
                            if corr_matrix.shape == (2, 2):
                                pearson_corr = float(corr_matrix[0, 1])
                                # Handle edge cases where correlation is NaN
                                if not np.isfinite(pearson_corr):
                                    pearson_corr = 0.0
                    except Exception:
                        pearson_corr = 0.0
                    
                    # calculate Spearman rank correlation coefficient
                    spearman_corr = 0.0
                    try:
                        if len(input_clean) > 1:
                            # use rank-based correlation to capture non-linear monotonic relationships
                            rank_input = stats.rankdata(input_clean)
                            rank_output = stats.rankdata(output_clean)
                            if np.std(rank_input) > 0 and np.std(rank_output) > 0:
                                spearman_matrix = np.corrcoef(rank_input, rank_output)
                                if spearman_matrix.shape == (2, 2):
                                    spearman_corr = float(spearman_matrix[0, 1])
                                    if not np.isfinite(spearman_corr):
                                        spearman_corr = 0.0
                    except Exception:
                        spearman_corr = 0.0
                    
                    # store correlation results for this input-output parameter pair
                    correlations.append({
                        'Input_Parameter': str(input_param),
                        'Pearson_Correlation': pearson_corr,
                        'Spearman_Correlation': spearman_corr,
                        'Abs_Pearson': abs(pearson_corr),
                        'Abs_Spearman': abs(spearman_corr),
                        'Sample_Size': len(input_clean),
                        'Excluded_Cases': min_length - len(input_clean)
                    })
                    
                except Exception as e:
                    # log warning for correlation calculation failures but continue processing
                    warnings.warn(f"Could not calculate correlation for {input_param} vs {output_param}: {e}")
                    continue
            
            # sort correlations by absolute Pearson correlation (desc)
            if correlations:
                corr_df = pd.DataFrame(correlations)
                corr_df = corr_df.sort_values('Abs_Pearson', ascending=False)
                sensitivity_results[output_param] = corr_df
            else:
                print(f"Skipped output parameter: {output_param} (no valid correlations)")
        
        print(f"Sensitivity analysis completed: {len(sensitivity_results)} output parameters analyzed")
        
        # store results in instance variable for later access
        self.sensitivity_results = sensitivity_results
        return sensitivity_results
    
    def _get_analyzable_cases_mask(self, results_df: pd.DataFrame) -> np.ndarray:
        """
        Get mask for cases that should be included in sensitivity analysis.
        Includes: normal cases + direct mode cases
        Excludes: parameter errors + COP failures
        """
        mask = np.ones(len(results_df), dtype=bool)
        
        # exclude parameter errors (physically invalid)
        if 'has_parameter_error' in results_df.columns:
            mask = mask & (~results_df['has_parameter_error'])
        
        # exclude COP failures (system unfeasible)
        if 'has_cop_failure' in results_df.columns:
            mask = mask & (~results_df['has_cop_failure'])
        
        # include both normal cases and direct mode cases
        # Direct mode is physically valid and represents optimal operation
        
        return mask
    
    def _preprocess_output_values(self, param_name: str, values: Any, results_df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess output values for sensitivity analysis.
        Handle special cases like infinite COPs in direct mode.
        """
        # Convert to numpy array to handle pandas series or other array-like types
        processed_values = np.asarray(values, dtype=np.float64).copy()
        
        # For COP parameters in direct mode, replace inf with a representative high value
        if 'cop' in param_name.lower() or 'ehp' in param_name.lower():
            # Find direct mode cases
            if param_name.startswith('heating_') and 'heating_direct_mode' in results_df.columns:
                direct_mode_mask = np.asarray(results_df['heating_direct_mode'].values, dtype=bool)
            elif param_name.startswith('cooling_') and 'cooling_direct_mode' in results_df.columns:
                direct_mode_mask = np.asarray(results_df['cooling_direct_mode'].values, dtype=bool)
            else:
                direct_mode_mask = np.zeros(len(processed_values), dtype=bool)
            
            # Replace inf values in direct mode with high representative value
            if np.any(direct_mode_mask):
                # Use 10x the maximum finite value, or 100 if no finite values exist
                finite_values = processed_values[np.isfinite(processed_values)]
                if len(finite_values) > 0:
                    replacement_value = float(np.max(finite_values) * 10)
                else:
                    replacement_value = 100.0
                
                inf_mask = np.isinf(processed_values)
                processed_values[inf_mask & direct_mode_mask] = replacement_value
        
        return processed_values

    def _log_case_breakdown(self, results_df: pd.DataFrame):
        """Log breakdown of different case types for transparency"""
        total_cases = len(results_df)
        
        normal_cases = 0
        direct_mode_cases = 0
        parameter_error_cases = 0
        cop_failure_cases = 0
        unknown_cases = 0
        
        if 'heating_anomaly_type' in results_df.columns and 'cooling_anomaly_type' in results_df.columns:
            # Count by examining both heating and cooling anomaly types
            for _, row in results_df.iterrows():
                heating_type = row['heating_anomaly_type']
                cooling_type = row['cooling_anomaly_type']
                
                # Prioritize most severe anomaly
                if heating_type == 'parameter_error' or cooling_type == 'parameter_error':
                    parameter_error_cases += 1
                elif heating_type == 'cop_failure' or cooling_type == 'cop_failure':
                    cop_failure_cases += 1
                elif heating_type == 'direct_mode' or cooling_type == 'direct_mode':
                    direct_mode_cases += 1
                elif heating_type == 'normal' and cooling_type == 'normal':
                    normal_cases += 1
                else:
                    unknown_cases += 1
        else:
            # Fallback to has_* columns
            if 'has_parameter_error' in results_df.columns:
                parameter_error_cases = int(results_df['has_parameter_error'].sum())
            if 'has_cop_failure' in results_df.columns:
                cop_failure_cases = int(results_df['has_cop_failure'].sum())
            if 'has_direct_mode' in results_df.columns:
                direct_mode_cases = int(results_df['has_direct_mode'].sum())
            
            # Estimate normal cases
            excluded_cases = parameter_error_cases + cop_failure_cases
            normal_cases = max(0, total_cases - excluded_cases - direct_mode_cases)
        
        analyzable_cases = normal_cases + direct_mode_cases
        
        print(f"Case breakdown for sensitivity analysis:")
        print(f"  Total cases: {total_cases}")
        print(f"  Normal cases: {normal_cases}")
        print(f"  Direct mode cases: {direct_mode_cases}")
        print(f"  Parameter error cases: {parameter_error_cases} (excluded)")
        print(f"  COP failure cases: {cop_failure_cases} (excluded)")
        print(f"  Unknown cases: {unknown_cases}")
        print(f"  Analyzable cases (normal + direct): {analyzable_cases}")
        print(f"  Analysis coverage: {analyzable_cases/total_cases*100:.1f}%")
        
        # Store breakdown for reporting
        self.case_breakdown = {
            'total_cases': total_cases,
            'normal_cases': normal_cases,
            'direct_mode_cases': direct_mode_cases,
            'parameter_error_cases': parameter_error_cases,
            'cop_failure_cases': cop_failure_cases,
            'unknown_cases': unknown_cases,
            'analyzable_cases': analyzable_cases,
            'analysis_coverage_percent': analyzable_cases/total_cases*100 if total_cases > 0 else 0
        }

    
    def get_parameter_importance_ranking(self) -> pd.DataFrame:
        """
        Get overall parameter importance ranking across all outputs
        """
        # guard
        if self.sensitivity_results is None:
            raise ValueError("No sensitivity results available. Run sensitivity analysis first.")
        # collect every input absolute Pearson correction for every output
        importance_scores: Dict[str, List[float]] = {}
        # iterate every output sensitivity df
        for output_param, sensitivity_df in self.sensitivity_results.items():
            for _, row in sensitivity_df.iterrows():
                input_param = str(row['Input_Parameter'])
                abs_corr = float(row['Abs_Pearson'])
                
                if input_param not in importance_scores:
                    importance_scores[input_param] = []
                importance_scores[input_param].append(abs_corr) # coz we dont care about the direction
        
        ranking_data: List[Dict[str, Any]] = []
        for input_param, scores in importance_scores.items():
            if len(scores) > 0:
                ranking_data.append({
                    'Parameter': input_param,
                    'Mean_Abs_Correlation': float(np.mean(scores)),
                    'Max_Abs_Correlation': float(np.max(scores)),
                    'Min_Abs_Correlation': float(np.min(scores)),
                    'Std_Abs_Correlation': float(np.std(scores)),
                    'Number_of_Outputs': int(len(scores))
                })
        
        if not ranking_data:
            return pd.DataFrame()
        # sort by DESC
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('Mean_Abs_Correlation', ascending=False)
        
        return ranking_df


    def export_results(self, filename_prefix: str = "ates_monte_carlo") -> Dict[str, str]:
        """
        Export Monte Carlo results to CSV files
        """
        # guard
        if self.results is None:
            raise ValueError("No results available. Run simulation first.")
        
        exported_files: Dict[str, str] = {}
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Export main results
        results_file = f"{filename_prefix}_results_{timestamp}.csv"
        self.results.to_csv(results_file, index=False)
        exported_files['results'] = results_file
        
        # Export statistics
        try:
            statistics = self.calculate_statistics()
            for group_name, stats_df in statistics.items():
                stats_file = f"{filename_prefix}_statistics_{group_name}_{timestamp}.csv"
                stats_df.to_csv(stats_file, index=False)
                exported_files[f'statistics_{group_name}'] = stats_file
        except Exception as e:
            print(f"Warning: Could not export statistics: {e}")
        
        # Export sensitivity analysis
        if self.sensitivity_results:
            for output_param, sensitivity_df in self.sensitivity_results.items():
                sens_file = f"{filename_prefix}_sensitivity_{output_param}_{timestamp}.csv"
                sensitivity_df.to_csv(sens_file, index=False)
                exported_files[f'sensitivity_{output_param}'] = sens_file
        
        return exported_files
    
# Utility functions for streamlit integration
def create_progress_callback(progress_bar, status_text):
    """
    Create a progress callback function for Streamlit
    """
    def callback(current: int, total: int) -> None:
        progress = current / total
        progress_bar.progress(progress)
        status_text.text(f"Completed {current}/{total} iterations ({progress:.1%})")
    return callback

def format_monte_carlo_summary(results_df: pd.DataFrame) -> str:
    """
    Format a summary of Monte Carlo results for display
    """
    if results_df is None or len(results_df) == 0:
        return "No results available"
    
    successful = int(results_df['success'].sum())
    total = len(results_df)
    success_rate = successful / total * 100
    
    summary = f"""**Monte Carlo Simulation Summary**
- Total iterations: {total:,}
- Successful calculations: {successful:,} ({success_rate:.1f}%)
- Failed calculations: {total - successful:,}"""
    
    if successful > 0:
        successful_df = results_df[results_df['success']]
        heating_cop_mean = float(successful_df['heating_system_cop'].mean())
        cooling_cop_mean = float(successful_df['cooling_system_cop'].mean())
        
        summary += f"""

**Key Results (Mean Values)**
- Heating System COP: {heating_cop_mean:.2f}
- Cooling System COP: {cooling_cop_mean:.2f}
"""
    
    return summary