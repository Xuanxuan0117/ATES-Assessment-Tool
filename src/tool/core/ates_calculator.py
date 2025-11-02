"""
ATES calculator based on spreadsheet
"""
import warnings
from dataclasses import dataclass 
from typing import Dict, Any

@dataclass
class ATESParameters:
    """
    ATES System Parameters class, including 25 input parameters(From spreadsheet D and G columns)
    and intermediate calculations
    """
    
    #A. Basic Physical Parameters
    aquifer_temp: float = 13.5                        # D3 - Taq Aquifer temperature (°C)
    water_density: float = 1000.0                     # D4 - rho_w Water density (kg/m³)
    water_specific_heat_capacity: float = 4184.0       # D5 - cp Water specific heat capacity (J/K/kg)
    thermal_recovery_factor: float = 0.4              # D8 - RT Thermal recovery factor (-)

    # B. System Operational Parameters
    heating_target_avg_flowrate_pd: float = 60.0      # D10 - qb,h Target average flow rate per doublet for heating (m³/hr)
    tolerance_in_energy_balance: float = 0.15         # D11 - εEBR Energy Balance Ratio tolerance (-)
    heating_number_of_doublets: int = 22              # D14 - nb Heating number of doublets (-)
    heating_months:float = 6.5                        # D17 - th Number of heating months (months)
    cooling_months: float = 3.5                       # D18 - tc Number of cooling months (months)
    pump_energy_density: float = 600.0                # D25 - Eg Pump energy density (kJ/m³)
    heating_ave_injection_temp: float = 10.0          # D27 - Ti,c Cold borehole(s) injection temperature(°C)
    heating_temp_to_building: float = 60.0            # D29 - Tb,h Building heating temperature (°C)

    # C. COP(coefficent of performance) Parameters
    cop_param_a: float = 100.0                        # D31 - a COP parameter a (-)
    cop_param_b: float = 1.6                          # D32 - b COP parameter b (-)
    cop_param_c: float = -0.08                        # D33 - c COP parameter c (-)
    cop_param_d: float = 7.0                          # D34 - d COP parameter d (-)
    carbon_intensity: float = 180.0                   # D36 - Ci Carbon intensity (gCO2/kWhe)

    # D. Cooling Side Paramters (G column)
    cooling_target_avg_flowrate_pd: float = 0.0       # G10 - qb,c Target average flow rate per doublet for cooling (m³/hr)
    cooling_number_of_doublets: int = 0               # G15 - nb Cooling number of doublets (-)
    cooling_total_produced_volume: float = 0.0        # G21 - Vp,c Total produced cooling volume (m³)
    cooling_ave_injection_temp: float = 21.0          # G27 - Ti,h Hot borehole(s) injection temperature(°C)
    cooling_temp_to_building: float = 14.0            # G29 - Tb,c Building cooling temperature (°C)

    # E. Auto-calculated Parameters
    water_volumetric_heat_capacity: float = 0.0       # D6  - cw Water volumetric heat capacity (J/K/m³)
    shoulder_months: float = 0.0                      # D19 - (-) Number of months which not heating and cooling (-)
    heating_total_produced_volume: float = 0.0        # D21 - Vp,h Total produced heating volume(m³)
    
    # turn on for direct calculation, off for monte-carlo
    _validation_enabled: bool = True

    def __post_init__(self):
         """
         Automatically called after __init__; computes derived parameters and validates inputs.
         """
         # D-Column auto-calculations
          # D6 = D4 * D5 
         self.water_volumetric_heat_capacity = self.water_density * self.water_specific_heat_capacity

          # D19 = 12-D17-D18
         self.shoulder_months = 12 - self.heating_months - self.cooling_months

         # G-column auto-calculations
          # G10 calculation(calculate cooling target average flowrate per doublet)
         self.calculate_g10()

          # G15 = D14 (cooling number of doublets = heating number of doublets)
         self.cooling_number_of_doublets = self.heating_number_of_doublets

         # D21&G21 Calculation(Total produced volume heating/cooling calculation)
         self.calculate_volumes()

         # conditional validation
         if ATESParameters._validation_enabled:
            self.validate_parameters()

    @classmethod
    def disable_validation(cls):
        """disable physical constraints for Mone Carlo simulation"""
        cls._validation_enabled = False
    
    @classmethod
    def enable_validation(cls):
        """enable physical constraints for Quick Look page"""
        cls._validation_enabled = True

    def calculate_g10(self):
        """
        Calculates G10 - Cooling flow rate per doublet.
        Formula: (1+D11)*D10*[(D3-D27)*D17-D8*(D27-D3)*D17]/[(D3-G27)*D18-D8*(G27-D3)*D18]
        """
        numerator = ((self.aquifer_temp - self.heating_ave_injection_temp) * self.heating_months -
                    self.thermal_recovery_factor * (self.heating_ave_injection_temp - self.aquifer_temp) * self.heating_months)

        denominator = ((self.aquifer_temp - self.cooling_ave_injection_temp) * self.cooling_months -
                      self.thermal_recovery_factor * (self.cooling_ave_injection_temp - self.aquifer_temp) * self.cooling_months)

        if abs(denominator) < 1e-10:
            raise ValueError("Denominator for G10 calculation is close to zero. Please check temperature parameters.")

        self.cooling_target_avg_flowrate_pd = -((1 + self.tolerance_in_energy_balance) *
                                           self.heating_target_avg_flowrate_pd *
                                           (numerator / denominator))
        
    def calculate_volumes(self):
        """
        Calculates D21 (Total produced heating volume) and G21 (Total produced cooling volume).
        """
        # calculate total flow rates(D14*D10)
        heating_total_flow = self.heating_number_of_doublets * self.heating_target_avg_flowrate_pd  # K6
        cooling_total_flow = self.cooling_number_of_doublets * self.cooling_target_avg_flowrate_pd  # N6

        # D21 = K6*D17*31*24
        self.heating_total_produced_volume = heating_total_flow * self.heating_months * 31 * 24

        # G21 = N6*D18*31*24
        self.cooling_total_produced_volume = cooling_total_flow * self.cooling_months * 31 * 24
    
    def validate_parameters(self):
        """
        Validate input parameters to ensure they are within reasonable and physically valid ranges.
        Raises ValueError for critical invalid inputs.
        """

        # A. Basic Physical Parameters 
        if not (0 <= self.aquifer_temp <= 100):
            raise ValueError(f"Aquifer temperature must be between 0 and 100 °C. Got {self.aquifer_temp}.")
        if self.water_density <= 0:
            raise ValueError(f"Water density must be positive. Got {self.water_density}.")
        if self.water_specific_heat_capacity <= 0:
            raise ValueError(f"Water specific heat capacity must be positive. Got {self.water_specific_heat_capacity}.")
        if not (0 <= self.thermal_recovery_factor <= 1):
            raise ValueError(f"Thermal recovery factor must be between 0 and 1. Got {self.thermal_recovery_factor}.")

        # B. System Operational Parameters 
        if self.heating_target_avg_flowrate_pd <= 0:
            raise ValueError(f"Heating target average flowrate per doublet must be positive. Got {self.heating_target_avg_flowrate_pd}.")
        
         # Heating number of doublets should be integer and non‑negative
        if not isinstance(self.heating_number_of_doublets, int):
            raise ValueError(f"Heating number of doublets must be an integer. Got {self.heating_number_of_doublets}.")
        if self.heating_number_of_doublets < 0:
            raise ValueError(f"Heating number of doublets cannot be negative. Got {self.heating_number_of_doublets}.")

         # Heating and cooling months
        if not (0 <= self.heating_months <= 12):
            raise ValueError(f"Heating months must be between 0 and 12. Got {self.heating_months}.")
        if not (0 <= self.cooling_months <= 12):
            raise ValueError(f"Cooling months must be between 0 and 12. Got {self.cooling_months}.")
        if self.heating_months + self.cooling_months > 12:
            raise ValueError(f"Sum of heating and cooling months cannot exceed 12. Got {self.heating_months + self.cooling_months}.")

        if self.pump_energy_density < 0:
            raise ValueError(f"Pump energy density cannot be negative. Got {self.pump_energy_density}.")

        if not (0 <= self.heating_ave_injection_temp <= 100):
            raise ValueError(f"Heating injection temperature must be between 0 and 100 °C. Got {self.heating_ave_injection_temp}.")
        if not (0 <= self.heating_temp_to_building <= 100):
            raise ValueError(f"Heating temperature to building must be between 0 and 100 °C. Got {self.heating_temp_to_building}.")

        # C. COP Parameters 
         # a should be > 0
        if self.cop_param_a < 0:
            raise ValueError(f"COP parameter a should not be negative. Got {self.cop_param_a}.")
         # b should be > 0
        if self.cop_param_b <= 0:
            raise ValueError(f"COP parameter 'b' must be > 0. Got {self.cop_param_b}.")
         # c usually should be <= 0, warn if not
        if self.cop_param_c > 0:
            import warnings
            warnings.warn(f"COP parameter 'c' is usually negative (COP decreases with ΔT). Got {self.cop_param_c}.")
         # d should be >= 0
        if self.cop_param_d < 0:
            raise ValueError(f"COP parameter 'd' must be >= 0. Got {self.cop_param_d}.")
         # carbon intensity should be >= 0
        if self.carbon_intensity < 0:
            raise ValueError(f"Carbon intensity cannot be negative. Got {self.carbon_intensity}.")

        # D. Cooling Side Parameters 
         #Cooling target average flowrate per doublet should be >= 0
        if self.cooling_target_avg_flowrate_pd < 0:
            raise ValueError(f"Cooling target average flowrate per doublet cannot be negative. Got {self.cooling_target_avg_flowrate_pd}.")
         # Cooling doublets should also be integer and >= 0
        if not isinstance(self.cooling_number_of_doublets, int):
            raise ValueError(f"Cooling number of doublets must be an integer. Got {self.cooling_number_of_doublets}.")
        if self.cooling_number_of_doublets < 0:
            raise ValueError(f"Cooling number of doublets cannot be negative. Got {self.cooling_number_of_doublets}.")

        # E. Derived parameters 
         # validate heating and cooling month based on the number of shoulder months
        if self.shoulder_months < 0:
            raise ValueError(f"Computed shoulder months is negative ({self.shoulder_months}). Check heating and cooling months.")  
         # total produced volume should be >= 0
        if self.heating_total_produced_volume < 0:
            raise ValueError(f"Total produced heating volume cannot be negative. Got {self.heating_total_produced_volume}.")
        if self.cooling_total_produced_volume < 0:
            raise ValueError(f"Total produced cooling volume cannot be negative. Got {self.cooling_total_produced_volume}.")

@dataclass
class ATESResults:
    """
    ATES Calculation Results class
    """

    #Heating output(K-Column with 32 parameters) 
    heating_total_energy_stored: float = 0.0             # K3 - (-) Total energy stored during heating (J)
    heating_stored_energy_recovered: float = 0.0         # K4 - (-) Stored energy recovered during heating (J)
    heating_total_flow_rate_m3hr: float = 0.0            # K6 - Vp Total flow rate during heating (m³/hr)
    heating_total_flow_rate_ls: float = 0.0              # K7 - Vp Total flow rate during heating (l/s)
    heating_total_flow_rate_m3s: float = 0.0             # K8 - Vp Total flow rate during heating (m³/s)
    heating_ave_production_temp: float = 0.0             # K10 - Tp Avegrage prodution temperature during heating (C)
    heating_ave_temp_change_across_HX: float = 0.0       # K11 - Tp-Ti Average temperature change across heat exchanger durinng heating (C)
    heating_temp_change_induced_HP: float = 0.0          # K12 - DT Temperature change induced by heat pump during heating (C)
    heating_heat_pump_COP: float = 0.0                   # K13 - COPhp Heat pump COP during heating (-)
    heating_ehp: float = 0.0                             # K14 - ehp Heat pump factor during heating (-) 
    heating_ave_power_to_HX_W: float = 0.0               # K16 - Pp,h Average power to heat exchanger(from aquifer) during heating (W)
    heating_ave_power_to_HX_MW: float = 0.0              # K17 - Pp,h Average power to heat exchanger(from aquifer) during heating (MW)
    heating_annual_energy_aquifer_J: float = 0.0         # K19 - (-) Annual energy produced from aquifer during heating (J)
    heating_annual_energy_aquifer_kWhth: float = 0.0     # K20 - (-) Annual energy produced from aquifer during heating (kWhth)
    heating_annual_energy_aquifer_GWhth: float = 0.0     # K21 - (-) Annual energy produced from aquifer during heating (GWhth)
    heating_monthly_to_HX: float = 0.0                   # K22 - (-) Monthly energy to heat exchanger(from aquifer) during heating (GWhth)
    energy_balance_ratio: float = 0.0                    # K23 - EBR Energy Balance Ratio (-)
    volume_balance_ratio: float = 0.0                    # K24 - VBR Volume Balance Ratio (-)
    heating_ave_power_to_building_W: float = 0.0         # K26 - Ps,h Average power to building during heating (W)
    heating_ave_power_to_building_MW: float = 0.0        # K27 - Ps,h Average power to building during heating (MW)
    heating_annual_energy_building_J: float = 0.0        # K29 - (-) Annual energy produced to building during heating (J)
    heating_annual_energy_building_kWhth: float = 0.0    # K30 - (-) Annual energy produced to building during heating (kWhth)
    heating_annual_energy_building_GWhth: float = 0.0    # K31 - (-) Annual energy produced to building during heating (GWhth)
    heating_monthly_to_building: float = 0.0             # K32 - Em,h Monthly energy to building during heating (GWhth)
    heating_elec_energy_hydraulic_pumps: float = 0.0     # K34 - (-) Electrical energy to control systems(hydraulic pump) during heating (J)
    heating_elec_energy_HP: float = 0.0                  # K35 - (-) Electrical energy to heat pump during heating (J)
    heating_annual_elec_energy_J: float = 0.0            # K36 - (-) Annual electrcial energy during heating (J)
    heating_annual_elec_energy_MWhe: float = 0.0         # K37 - (-) Annual electrcial energy during heating (MWhe)
    heating_annual_elec_energy_GWhe: float = 0.0         # K38 - (-) Annual electrcial energy during heating (GWhe)
    heating_system_cop: float = 0.0                      # K39 - COPs System Coefficent of Performance during heating
    heating_elec_energy_per_thermal: float = 0.0         # K40 - Es/e Electrical energy required per unit heating energy supplied by the system（kWhe/kWhth)
    heating_co2_emissions_per_thermal: float = 0.0       # K41 - C CO2 emitted per unit heating supplied (gCO2/kWhth)
    
    #Cooling output(N-Column with 30 parameters) 
    cooling_total_energy_stored: float = 0.0             # N3 - (-) Total energy stored during cooling (J)
    cooling_stored_energy_recovered: float = 0.0         # N4 - (-) Stored energy recovered during cooling (J)
    cooling_target_avg_flowrate_pd: float = 0.0          # G10 - qb,c Target average flow rate per doublet for cooling (m³/hr)
    cooling_total_flow_rate_m3hr: float = 0.0            # N6 - Vp Total flow rate during cooling (m³/hr)
    cooling_total_flow_rate_ls: float = 0.0              # N7 - Vp Total flow rate during cooling (l/s)
    cooling_total_flow_rate_m3s: float = 0.0             # N8 - Vp Total flow rate during cooling (m³/s)
    cooling_ave_production_temp: float = 0.0             # N10 - Tp Avegrage prodution temperature during cooling (C)
    cooling_ave_temp_change_across_HX: float = 0.0       # N11 - Tp-Ti Average temperature change across heat exchanger durinng cooling (C)
    cooling_temp_change_induced_HP: float = 0.0          # N12 - DT Temperature change induced by heat pump during cooling (C)
    cooling_heat_pump_COP: float = 0.0                   # N13 - COPhp Heat pump COP during cooling (-)
    cooling_ehp: float = 0.0                             # N14 - ehp Heat pump factor during cooling (-) 
    cooling_ave_power_to_HX_W: float = 0.0               # N16 - Pp,c Average power to heat exchanger(from aquifer) during cooling (W)
    cooling_ave_power_to_HX_MW: float = 0.0              # N17 - Pp,c Average power to heat exchanger(from aquifer) during cooling (MW)
    cooling_annual_energy_aquifer_J: float = 0.0         # N19 - (-) Annual energy produced from aquifer during cooling (J)
    cooling_annual_energy_aquifer_kWhth: float = 0.0     # N20 - (-) Annual energy produced from aquifer during cooling (kWhth)
    cooling_annual_energy_aquifer_GWhth: float = 0.0     # N21 - (-) Annual energy produced from aquifer during cooling (GWhth)
    cooling_monthly_to_HX: float = 0.0                   # N22 - (-) Monthly energy to heat exchanger(from aquifer) during cooling (GWhth)
    cooling_ave_power_to_building_W: float = 0.0         # N26 - Ps,c Average power to building during cooling (W)
    cooling_ave_power_to_building_MW: float = 0.0        # N27 - Ps,c Average power to building during cooling (MW)
    cooling_annual_energy_building_J: float = 0.0        # N29 - (-) Annual energy produced to building during cooling (J)
    cooling_annual_energy_building_kWhth: float = 0.0    # N30 - (-) Annual energy produced to building during cooling (kWhth)
    cooling_annual_energy_building_GWhth: float = 0.0    # N31 - (-) Annual energy produced to building during cooling (GWhth)
    cooling_monthly_to_building: float = 0.0             # N32 - Em,h Monthly energy to building during cooling (GWhth)
    cooling_elec_energy_hydraulic_pumps: float = 0.0     # N34 - (-) Electrical energy to control systems(hydraulic pump) during cooling (J)
    cooling_elec_energy_HP: float = 0.0                  # N35 - (-) Electrical energy to heat pump during cooling (J)
    cooling_annual_elec_energy_J: float = 0.0            # N36 - (-) Annual electrcial energy during cooling (J)
    cooling_annual_elec_energy_MWhe: float = 0.0         # N37 - (-) Annual electrcial energy during cooling (MWhe)
    cooling_annual_elec_energy_GWhe: float = 0.0         # N38 - (-) Annual electrcial energy during cooling (GWhe)
    cooling_system_cop: float = 0.0                      # N39 - COPs System Coefficent of Performance during heating
    cooling_elec_energy_per_thermal: float = 0.0         # N40 - Es/e Electrical energy required per unit cooling energy supplied by the system（kWhe/kWhth)
    cooling_co2_emissions_per_thermal: float = 0.0       # N41 - C CO2 emitted per unit cooling supplied (gCO2/kWhth)

    # identify Direct Mode(whether we are using heat pump to heat/cool)
    heating_direct_mode: bool = False
    cooling_direct_mode: bool = True

class ATESCalculator:
    """
    ATES System Calculator
    """

    def __init__(self, parameters: ATESParameters):
        """
        Initializes the ATESCalculator with given parameters.

        Arg:
            parameters (ATESParameters): An instance of ATESParameters containing all input data.
        """
        self.params = parameters
        self.results = ATESResults()

    def calculate(self) -> ATESResults:
        """
        Executes the full calculation sequence

        Returns:
        ATESResults: An instance of ATESResults containing all calculated output prameters.
        """
        # calculate heating side ouput (K column)
        self._calculate_heating_outputs()

        # calculate cooling side output (N column)
        self._calculate_cooling_outputs()

        # calculate balance ratios
        self._calculate_balance_ratios()

        return self.results
    
    def _calculate_heating_outputs(self):
        """
        Calculate all heating-column outputs with direct mode logic
        """
        p = self.params
        r = self.results

        # K3 = $D6*G21*(G27-D3)
        r.heating_total_energy_stored = (p.water_volumetric_heat_capacity *
                                  p.cooling_total_produced_volume *
                                  (p.cooling_ave_injection_temp - p.aquifer_temp))
        
        # K4 = K3*D8
        r.heating_stored_energy_recovered = r.heating_total_energy_stored * p.thermal_recovery_factor

        # K6 = D14*D10
        r.heating_total_flow_rate_m3hr = p.heating_number_of_doublets * p.heating_target_avg_flowrate_pd

        # K7 = K6*0.27777777778
        r.heating_total_flow_rate_ls = r.heating_total_flow_rate_m3hr * 0.27777777778

        # K8 = K6/60/60
        r.heating_total_flow_rate_m3s = r.heating_total_flow_rate_m3hr / 3600
        # calulate physical heating temperatur and decide based on different mode
        calculated_physical_heating_temp = (p.aquifer_temp +
                                p.thermal_recovery_factor *
                                (p.cooling_ave_injection_temp - p.aquifer_temp))

        # K10 = calculated_physical_temp
        r.heating_ave_production_temp = calculated_physical_heating_temp
        # check if production temperature meets building requirement during heating
        # Direct mode when Tp >= Tb,h (production temp >= building requirement)
        calculated_physical_heating_temp = r.heating_ave_production_temp  
        if calculated_physical_heating_temp >= p.heating_temp_to_building:
            # Direct heating mode (production temperature is sufficient)
            r.heating_direct_mode = True
            r.heating_heat_pump_COP = float('inf')
            r.heating_ehp = 1.0 # direct transfer
            
            # use building target temperature to recalculate
            r.heating_ave_production_temp = p.heating_temp_to_building
            r.heating_ave_temp_change_across_HX = r.heating_ave_production_temp - p.heating_ave_injection_temp
            r.heating_temp_change_induced_HP = 0.0 # no heat pump needed
            
            warnings.warn(f"Direct heating mode: Physical temp ({calculated_physical_heating_temp:.1f}°C) >= "
                        f"Building requirement ({p.heating_temp_to_building:.1f}°C). "
                        f"System provides exactly {p.heating_temp_to_building:.1f}°C to building.")
        else:
            # Heating pump heating mode for boost temperature
            r.heating_direct_mode = False

            ## K11 = K10-D27 (for heat exchanger)
            r.heating_ave_temp_change_across_HX = r.heating_ave_production_temp - p.heating_ave_injection_temp

            # K12 = D29-K10 (temperature boost needed by heat pump)
            r.heating_temp_change_induced_HP = p.heating_temp_to_building - r.heating_ave_production_temp

            if r.heating_temp_change_induced_HP <= 0:
                raise ValueError(f"Heating temperature difference must be > 0, current value: {r.heating_temp_change_induced_HP}")
            
            # K13 = $D31*(1/K12^$D32)+$D33*K12+D34
            r.heating_heat_pump_COP = (p.cop_param_a * (1 / (r.heating_temp_change_induced_HP ** p.cop_param_b)) +
                            p.cop_param_c * r.heating_temp_change_induced_HP +
                            p.cop_param_d)
            
            # K14 = K13/(K13-1)
            if r.heating_heat_pump_COP <= 1:
                warnings.warn(f"Heating COP ({r.heating_heat_pump_COP:.2f}) <= 1.")
                r.heating_ehp = float('inf')
            else:
                r.heating_ehp = r.heating_heat_pump_COP / (r.heating_heat_pump_COP - 1)

        # K19 = (D6*D21*(D3-D27))+(D8*D6*G21*(G27-D3))
        r.heating_annual_energy_aquifer_J = ((p.water_volumetric_heat_capacity *
                                        p.heating_total_produced_volume *
                                        (p.aquifer_temp - p.heating_ave_injection_temp)) +
                                        (p.thermal_recovery_factor *
                                        p.water_volumetric_heat_capacity *
                                        p.cooling_total_produced_volume *
                                        (p.cooling_ave_injection_temp - p.aquifer_temp)))

        # K16 = K19/D17/31/24/60/60
        r.heating_ave_power_to_HX_W = (r.heating_annual_energy_aquifer_J /p.heating_months / 31 / 24 / 3600)

        # K17 = K16/1000000
        r.heating_ave_power_to_HX_MW= r.heating_ave_power_to_HX_W / 1000000


        # K20 = K19/3600000
        r.heating_annual_energy_aquifer_kWhth = r.heating_annual_energy_aquifer_J/ 3600000

        # K21 = K20/1000000
        r.heating_annual_energy_aquifer_GWhth = r.heating_annual_energy_aquifer_kWhth / 1000000

        # K22 = K21/D17
        r.heating_monthly_to_HX = r.heating_annual_energy_aquifer_GWhth / p.heating_months

        # Building energy calculations
        if r.heating_direct_mode:
            # Direct heating: building gets energy at production temperature
            r.heating_ave_power_to_building_W = r.heating_ave_power_to_HX_W
            r.heating_annual_energy_building_J = r.heating_annual_energy_aquifer_J
        else:
            # Heat pump heating: apply heat pump factor
            # K26 = K16*K14
            r.heating_ave_power_to_building_W = r.heating_ave_power_to_HX_W * r.heating_ehp
            # K29 = K14*K19
            r.heating_annual_energy_building_J = r.heating_annual_energy_aquifer_J * r.heating_ehp

        # K27 = K26/1000000
        r.heating_ave_power_to_building_MW = r.heating_ave_power_to_building_W / 1000000

        # K30 = K29/3600000
        r.heating_annual_energy_building_kWhth = r.heating_annual_energy_building_J / 3600000

        # K31 = K30/1000000
        r.heating_annual_energy_building_GWhth = r.heating_annual_energy_building_kWhth / 1000000

        # K32 = K31/D17
        r.heating_monthly_to_building = r.heating_annual_energy_building_GWhth / p.heating_months

        # Electrical energy calculations
        # K34 = D25*D21*1000
        r.heating_elec_energy_hydraulic_pumps = p.pump_energy_density * p.heating_total_produced_volume * 1000

        if r.heating_direct_mode:
            # direct heating: no heat pump electrical energy
            r.heating_elec_energy_HP = 0
        else:
            # K35 = K29/K13
            r.heating_elec_energy_HP = r.heating_annual_energy_building_J / r.heating_heat_pump_COP

        # K36 = K34+K35
        r.heating_annual_elec_energy_J = (r.heating_elec_energy_hydraulic_pumps + r.heating_elec_energy_HP)

        # K37 = K36/3600000000
        r.heating_annual_elec_energy_MWhe = r.heating_annual_elec_energy_J / 3600000000

        # K38 = K37/1000
        r.heating_annual_elec_energy_GWhe = r.heating_annual_elec_energy_MWhe / 1000

        # K39 = K31/K38
        if r.heating_annual_elec_energy_GWhe > 0:
            r.heating_system_cop = r.heating_annual_energy_building_GWhth / r.heating_annual_elec_energy_GWhe
        else:
            r.heating_system_cop = float('inf')

        # K40 = 1/K39
        if r.heating_system_cop > 0 and r.heating_system_cop != float('inf'):
            r.heating_elec_energy_per_thermal = 1 / r.heating_system_cop
        else:
            r.heating_elec_energy_per_thermal = 0

        # K41 = D36*K40
        r.heating_co2_emissions_per_thermal = p.carbon_intensity * r.heating_elec_energy_per_thermal

    def _calculate_cooling_outputs(self):
        """
        Calculate all cooling-column outputs with direct mode logic
        """
        p = self.params
        r = self.results

        # Store the cooling target flow rate from parameters
        r.cooling_target_avg_flowrate_pd = p.cooling_target_avg_flowrate_pd

        # N3 = D6*D21*(D3-D27)
        r.cooling_total_energy_stored = (
            p.water_volumetric_heat_capacity *
            p.heating_total_produced_volume *
            (p.aquifer_temp - p.heating_ave_injection_temp)
        )

        # N4 = N3*D8
        r.cooling_stored_energy_recovered = r.cooling_total_energy_stored * p.thermal_recovery_factor

        # N6 = G15*G10
        r.cooling_total_flow_rate_m3hr = p.cooling_number_of_doublets * p.cooling_target_avg_flowrate_pd

        # N7 = N6*0.27777777778
        r.cooling_total_flow_rate_ls = r.cooling_total_flow_rate_m3hr * 0.27777777778

        # N8 = N6/60/60
        r.cooling_total_flow_rate_m3s = r.cooling_total_flow_rate_m3hr / 3600

    # calculate the groundwater temperature
        calculated_physical_cooling_temp = (
            p.aquifer_temp +
            p.thermal_recovery_factor * (p.heating_ave_injection_temp - p.aquifer_temp)
        )

        # decide whether we are using direct cooling mode
        if calculated_physical_cooling_temp <= p.cooling_temp_to_building:
            # direct cooling
            r.cooling_direct_mode = True
            r.cooling_heat_pump_COP = float('inf')
            r.cooling_ehp = 1.0
            r.cooling_temp_change_induced_HP = 0.0
            
            # building required temperature
            r.cooling_ave_production_temp = p.cooling_temp_to_building 
            
            # N11 = G27 - N10 (Average temperature change across heat exchanger during cooling)
            r.cooling_ave_temp_change_across_HX = p.cooling_ave_injection_temp - r.cooling_ave_production_temp
            
            warnings.warn(
                f"Direct cooling mode: Physical temp ({calculated_physical_cooling_temp:.1f}°C) <= "
                f"Building requirement ({p.cooling_temp_to_building:.1f}°C). "
                f"System provides exactly {p.cooling_temp_to_building:.1f}°C to building."
            )
        else:
            # Need to use heat pump to cool down the water
            r.cooling_direct_mode = False
            
            # N10 = groundwater temperature
            r.cooling_ave_production_temp = calculated_physical_cooling_temp

            # N11 = G27 - N10 (for heat exchanger)
            r.cooling_ave_temp_change_across_HX = p.cooling_ave_injection_temp - r.cooling_ave_production_temp

            # N12 = N10 - G29 (temperature reduction needed by heat pump)
            r.cooling_temp_change_induced_HP = r.cooling_ave_production_temp - p.cooling_temp_to_building

            if r.cooling_temp_change_induced_HP <= 0:
                raise ValueError(
                    f"Cooling temperature difference must be > 0, current value: {r.cooling_temp_change_induced_HP}"
                )

            # N13 = D31*(1/N12^$D32)+$D33*N12+D34
            r.cooling_heat_pump_COP = (
                p.cop_param_a * (1 / (r.cooling_temp_change_induced_HP ** p.cop_param_b)) +
                p.cop_param_c * r.cooling_temp_change_induced_HP +
                p.cop_param_d
            )

            # N14 = N13/(N13-1)
            if r.cooling_heat_pump_COP <= 1:
                warnings.warn(f"Cooling COP ({r.cooling_heat_pump_COP:.2f}) <= 1.")
                r.cooling_ehp = float('inf')
            else:
                r.cooling_ehp = r.cooling_heat_pump_COP / (r.cooling_heat_pump_COP - 1)

        # N19 = (D6*G21*(D3-G27))+(D8*D6*D21*(D27-D3))
        r.cooling_annual_energy_aquifer_J = (p.water_volumetric_heat_capacity * p.cooling_total_produced_volume* (p.cooling_ave_injection_temp - p.aquifer_temp)
                                             + p.thermal_recovery_factor * p.water_volumetric_heat_capacity * p.heating_total_produced_volume *(p.aquifer_temp 
                                                                                                                                       - p.heating_ave_injection_temp))


        # N16 = N19/D18/31/24/60/60
        r.cooling_ave_power_to_HX_W = r.cooling_annual_energy_aquifer_J / p.cooling_months / 31 / 24 / 3600

        # N17 = N16/1000000
        r.cooling_ave_power_to_HX_MW = r.cooling_ave_power_to_HX_W / 1_000_000

        # N20 = N19/3600000
        r.cooling_annual_energy_aquifer_kWhth = r.cooling_annual_energy_aquifer_J / 3_600_000

        # N21 = N20/1000000
        r.cooling_annual_energy_aquifer_GWhth = r.cooling_annual_energy_aquifer_kWhth / 1_000_000

        # N22 = N21/D18
        r.cooling_monthly_to_HX = r.cooling_annual_energy_aquifer_GWhth / p.cooling_months

        # building energy calculations
        if r.cooling_direct_mode:
            # Direct cooling: building gets energy at production temperature
            r.cooling_ave_power_to_building_W = r.cooling_ave_power_to_HX_W
            r.cooling_annual_energy_building_J = r.cooling_annual_energy_aquifer_J
        else:
            # Heat pump cooling: apply heat pump factor
            # N26 = N16*N14
            r.cooling_ave_power_to_building_W = r.cooling_ave_power_to_HX_W * r.cooling_ehp
            # N29 = N14*N19
            r.cooling_annual_energy_building_J = r.cooling_ehp * r.cooling_annual_energy_aquifer_J

        # N27 = N26/1000000
        r.cooling_ave_power_to_building_MW = r.cooling_ave_power_to_building_W / 1_000_000

        # N30 = N29/3600000
        r.cooling_annual_energy_building_kWhth = r.cooling_annual_energy_building_J / 3_600_000

        # N31 = N30/1000000
        r.cooling_annual_energy_building_GWhth = r.cooling_annual_energy_building_kWhth / 1_000_000

        # N32 = N31/D18
        r.cooling_monthly_to_building = r.cooling_annual_energy_building_GWhth / p.cooling_months

        # Electrical energy calculations
        # N34 = D25*G21*1000
        r.cooling_elec_energy_hydraulic_pumps = p.pump_energy_density * p.cooling_total_produced_volume * 1000

        if r.cooling_direct_mode:
            # Direct cooling: no heat pump electrical energy
            r.cooling_elec_energy_HP = 0
        else:
            # N35 = N29/N13
            r.cooling_elec_energy_HP = r.cooling_annual_energy_building_J / r.cooling_heat_pump_COP

        # N36 = N34+N35
        r.cooling_annual_elec_energy_J = r.cooling_elec_energy_hydraulic_pumps + r.cooling_elec_energy_HP

        # N37 = N36/3600000000
        r.cooling_annual_elec_energy_MWhe = r.cooling_annual_elec_energy_J / 3_600_000_000

        # N38 = N37/1000
        r.cooling_annual_elec_energy_GWhe = r.cooling_annual_elec_energy_MWhe / 1000

        # N39 = N31/N38
        if r.cooling_annual_elec_energy_GWhe > 0:
            r.cooling_system_cop = r.cooling_annual_energy_building_GWhth / r.cooling_annual_elec_energy_GWhe
        else:
            r.cooling_system_cop = float('inf')

        # N40 = 1/N39
        if r.cooling_system_cop > 0 and r.cooling_system_cop != float('inf'):
            r.cooling_elec_energy_per_thermal = 1 / r.cooling_system_cop
        else:
            r.cooling_elec_energy_per_thermal = 0

        # N41 = D36*N40
        r.cooling_co2_emissions_per_thermal = p.carbon_intensity * r.cooling_elec_energy_per_thermal

    def _calculate_balance_ratios(self):
        """
        Calculate EBR and VBR after both heating and cooling calculations are complete
        """
        p = self.params
        r = self.results
        
        # K23 - Energy Balance Ratio (EBR)
        # EBR = (N19 - K19) / (K19 + N19)
        total_energy = r.heating_annual_energy_aquifer_J + r.cooling_annual_energy_aquifer_J
        if total_energy > 0:
            r.energy_balance_ratio = (r.cooling_annual_energy_aquifer_J - r.heating_annual_energy_aquifer_J) / total_energy
        else:
            r.energy_balance_ratio = 0.0

        # K24 - Volume Balance Ratio (VBR)
        # VBR = (D21 - G21) / (D21 + G21)
        total_volume = p.heating_total_produced_volume + p.cooling_total_produced_volume
        if total_volume > 0:
            r.volume_balance_ratio = (p.heating_total_produced_volume - p.cooling_total_produced_volume) / total_volume
        else:
            r.volume_balance_ratio = 0.0

    def get_mode_info(self) -> Dict[str, Any]:
        """
        Returns information about both heating and cooling modes.
        
        Returns:
            Dict[str, Any]: mode status, temperatures, and diagnostics.
        """
        return {
            # Heating mode info
            'heating_direct_mode': self.results.heating_direct_mode,
            'heating_needs_heat_pump': not self.results.heating_direct_mode,
            'heating_production_temp': self.results.heating_ave_production_temp,
            'heating_building_temp': self.params.heating_temp_to_building,
            'heating_temp_difference': self.params.heating_temp_to_building - self.results.heating_ave_production_temp,

            # Cooling mode info
            'cooling_direct_mode': self.results.cooling_direct_mode,
            'cooling_needs_heat_pump': not self.results.cooling_direct_mode,
            'cooling_production_temp': self.results.cooling_ave_production_temp,
            'cooling_building_temp': self.params.cooling_temp_to_building,
            'cooling_temp_difference': self.results.cooling_ave_production_temp - self.params.cooling_temp_to_building,
        }

if __name__ == "__main__":
    # simple test run
    params = ATESParameters()
    calculator = ATESCalculator(params)
    results = calculator.calculate()
    
    print("ATES Calculator Test Run:")
    print(f"Heating System COP: {results.heating_system_cop:.2f}")
    print(f"Cooling System COP: {'Direct Mode' if results.cooling_system_cop == float('inf') else f'{results.cooling_system_cop:.2f}'}")
    print(f"Heating Direct Mode: {results.heating_direct_mode}")
    print(f"Cooling Direct Mode: {results.cooling_direct_mode}")
    print("test completed successfully!")