from typing import Dict
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
eps = np.finfo(float).eps

#  =========== FIXED & CALCULATED CONSTANTS ================================
class Struct(): pass

constants = Struct()
# Fixed constants:
constants.Mwglc = 180                                                                # Molweight glucose (g/mol)
constants.Mwnh3 = 17                                                                 # Molweight NH3 (g/mol)
constants.Mwx = 26                                                                   # Molweight biomass (g/mol), although 24.6 (g/mol) according formula: C1 H1.8 O0.5 N0.2
constants.Mwp = 24                                                                   # Molweight protein (g/mol), although 22.6 (g/mol) according formula: C1 H1.6 O0.3 N0.3
constants.Mwo2 = 32                                                                  # Molweight O2 (g/mol)
constants.Mwco2 = 44                                                                 # Molweight CO2 (g/mol)
constants.Mwethd = 46.069                                                            # Molweight Ethanol (g/mol)
constants.yo2_air = 0.21                                                             # O2 fraction ingas (mol/mol)
constants.yco2_air = 0.0004                                                          # CO2 fraction ingas (mol/mol)

# Feed solutions:
constants.Cglcf = 0.300                                                              # Glucose concentration in feed solution (kg/kg)
constants.Cnh3t = 0.25                                                               # NH3 concentration in titrant (kg/kg)

# Kinetics:
constants.q1max = 0.1                                                                # Maximum specific growth rate (1/h)
constants.K1 = 0.1                                                                   # Affinity growth for glc (g/kg)
constants.K2 = 0.1                                                                   # Affinity growth for nh3 (g/kg)
constants.K3 = 0.1                                                                   # Ratio protein formation rate to growth (-)
constants.K4 = 2                                                                     # Decline of protein formation towards max growth (-)
constants.K5 = -100                                                                  # Onset decline of protein formation rate (kg/g)
constants.K6 = 0.1                                                                   # Onset decline of protein formation rate (g/kg)
constants.q4max = 4.5                                                                # Maximum maintenance (kJ/Cmol X/h)
constants.K7 = 0.01                                                                  # Affinity maintenance for glc (g/kg)
constants.G1 = 240                                                                   # Energy consumption for growth (kJ/cmol X)
constants.G2 = 220                                                                   # Energy consumption for protein formation (kJ/Cmol p)
constants.G3 = 2843                                                                  # Energy generation by glc catabolism (kJ/mol glc)

  
def call_fermentation_model(settings):
    """
    Gets all the settings, initial conditions and constants, and does the ODE integration.
    Returns just the bare integration results.
    """
    
    # Defaults are "Full batch operation": no in or out flow of anything, 
    #.             no titrant, no glucose, no product withdrawal.
    defaults = {
        "feed_glucose": False, # Indicates to never feed glucose
        "Cglc0": 50,     # Initial glucose concentration (g/kg) in the vessel
        "Cnh30": 3.0,    # Initial NH3 concentration (g/kg)
        "Cx0": 1,        # Initial biomass concentration (g/kg) [inoculation]
        "evap":  0,      # Evaporation rate (g/h)
        "WB0":  10,      # Initial broth weight (kg) in the vessel 
        "Duration": 66,      # Duration of fermentation (h)
    }
    
    # But then update the parts of the settings which override this
    defaults.update(settings)
    conditions, initials = get_conditions_initials(settings)
    
    # Then do the simulation
    return solve_ivp(
        diffeq,
        t_span=(0, conditions.Tend),
        y0=[initials.WB0, initials.Mglc0, initials.Mnh30, initials.Mx0, initials.Mp0, initials.Tglc0, initials.Tnh30, initials.To20, initials.Tco20, initials.Th2o0],
        t_eval=np.arange(start=0, step=conditions.Tint, stop=conditions.Tend),
        args=(constants, conditions),
        method="RK23",
    )

def get_conditions_initials(settings: Dict):
    class Struct(): pass
    conditions = Struct()

    conditions.Tend = settings["Duration"]
    conditions.Tint = 0.25                                                              # Time resolution of simulation (h)
    conditions.airflow = 10                                                             # Airflow rate (Nl/min)
    conditions.Fglc0  = 3.0                                                             # Initial glucose feed rate (g/h)
    conditions.expF = 0.07                                                              # Exponential increase of feed rate (1/h)
    conditions.slopeF = 1.0                                                             # Slope of feed rate (g/h/h), if feeding linearly.
    conditions.feed_profile = "exponential" #settings["feed_profile"]
    conditions.temperature = 25                                                         # Fermentation T (C)
    conditions.evap = settings["Evaporation"]
    conditions.feed_glucose = bool(settings["C glc t0"])
    conditions.flagtime= 0
    assert conditions.feed_profile in ["linear", "exponential"]
    
    
    initials = Struct()
    initials.Cglc0 = settings["C glc t0"]                                               # Initial glucose concentration (g/kg)
    initials.Cnh30 = settings["C NH3 t0"]                                               # Initial NH3 concentration (g/kg)
    initials.Cx0 = settings["C X t0"]                                                   # Initial biomass concentration (g/kg)
    initials.WB0 = settings["WB t0"]                                                    # Initial broth weight (kg) 
    initials.Cp0 = 0                                                                    # Initial protein concentration (g/kg)
    initials.Mglc0 = initials.Cglc0*initials.WB0                                        # Initial glucose weight (g)
    initials.Mnh30 = initials.Cnh30*initials.WB0                                        # Initial NH3 weight (g)
    initials.Mx0 = initials.Cx0*initials.WB0                                            # Initial biomass weight (g)
    initials.Mp0 = initials.Cp0*initials.WB0                                            # Initial protein weight (g)
    initials.Tglc0 = initials.Mglc0                                                     # Totalized glucose dosed (g)
    initials.Tnh30 = initials.Mnh30                                                     # Totalized NH3 dosed (g)
    initials.To20 = 0                                                                   # Totalized O2 consumed (mol)
    initials.Tco20 = 0                                                                  # Totalized CO2 produced (mol)
    initials.Th2o0 = 0                                                                  # Totalized H2O evaporated (g)
    
    return conditions, initials
    
  

def complete_balance_calculations(profiles, settings):
    """
    `profiles` is the solution from the solve_ipv function.
    """
    conditions, initials = get_conditions_initials(settings)
    columns = [
        "Broth weight [kg broth]", 
        "Mglc [g glucose]", 
        "Mnh3 [g NH3]", 
        "Mx [g biomass X]", 
        "Mp [g protein p]", 
        "Σ glucose feed [g glucose]", 
        "Σ NH3 feed [kg NH3]", 
        "Σ O2 [mol O2]", 
        "Σ CO2 [mol CO2]", 
        "Σ evap H2O [g H2O]"
    ]
    outputs = pd.DataFrame(profiles.y).T
    outputs.columns = columns
    outputs["Time [hours]"] = profiles.t

    extra_columns = []
    for _, values in outputs.iterrows():    
        addition = {}
        c, v, y = kineq(values["Time [hours]"], values.values, const=constants, conditions=conditions)

        addition["Cglc [g/kg]"], addition["Cnh3 [g/kg]"], addition["Cx [g/kg]"], addition["Cp g/kg]"] = c
        addition['Qglc [g/g X/h]'],addition['Qnh3 [g/g X/h]'],addition['Qo2 [mol/g X/h]'],addition['Qx [1/h]'],addition['Qp [g/g X/h]'],addition['Qco2 [mol/g X/h]'],addition['Fglc'],addition['Fnh3'],addition["OUR [mmol/kg/h]"],addition["CPR [mmol/kg/h]"] = v
        addition['yO2 [mol/mol]'],addition['yCO2 [mol/mol]'],addition['Airflow [NL/min]'],addition['Offgas flow [NL/min]'] = y
        addition["Time [hours]"] = values["Time [hours]"]
        assert (
            len([v for v in addition.values() if np.isnan(v)]) == 0
        ), f"There are missing values at {addition['Time [hours]']} hours"
        extra_columns.append(addition)

    # Merge datasets, based on the time values    
    all_outputs = outputs.set_index("Time [hours]").join(pd.DataFrame(extra_columns).set_index("Time [hours]"))
    # But then reset the index, to allow further calculations
    all_outputs = all_outputs.reset_index()
    all_outputs["Rs [g/h]"] = all_outputs['Qglc [g/g X/h]'] * all_outputs["Mx [g biomass X]"]
    all_outputs["Rnh3 [g/h]"] = all_outputs['Qnh3 [g/g X/h]'] * all_outputs["Mx [g biomass X]"]
    all_outputs["Rx [g/h]"] = all_outputs['Qx [1/h]'] * all_outputs["Mx [g biomass X]"]
    all_outputs["Rp [g/h]"] = all_outputs['Qp [g/g X/h]'] * all_outputs["Mx [g biomass X]"]
    all_outputs["RQ [-]"] = all_outputs["CPR [mmol/kg/h]"] / all_outputs["OUR [mmol/kg/h]"]
    all_outputs["Yxs [g/g]"] = ( all_outputs["Mx [g biomass X]"] - initials.Mx0 )/ (all_outputs["Σ glucose feed [g glucose]"] - all_outputs["Mglc [g glucose]"])
    all_outputs["Yps [g/g]"] = ( all_outputs["Mp [g protein p]"] - initials.Mp0 )/ (all_outputs["Σ glucose feed [g glucose]"] - all_outputs["Mglc [g glucose]"])
    all_outputs["Yos [g/g]"] = ( all_outputs["Σ O2 [mol O2]"]  )/ (all_outputs["Σ glucose feed [g glucose]"] - all_outputs["Mglc [g glucose]"]) * 1000
    all_outputs["Prod [g/kg/h]"] = all_outputs["Mp [g protein p]"] / all_outputs["Broth weight [kg broth]"] / all_outputs["Time [hours]"]
    
    # Force small values to zero
    for colname, column in all_outputs.iteritems():        
        all_outputs.loc[column.abs() < np.sqrt(eps), colname] = 0        

    # Mass balance [grams]:
    mass_in = initials.WB0*1000 + \
              (all_outputs["Σ glucose feed [g glucose]"] - initials.Mglc0)/constants.Cglcf + \
              (all_outputs["Σ NH3 feed [kg NH3]"] - initials.Mnh30)/constants.Cnh3t +\
              (all_outputs["Σ O2 [mol O2]"] * constants.Mwo2)
    mass_out = all_outputs["Broth weight [kg broth]"]*1000 + \
               all_outputs["Σ CO2 [mol CO2]"] * constants.Mwco2 + \
               all_outputs["Σ evap H2O [g H2O]"]
    mass_balance_relgap = (mass_in - mass_out)/mass_in * 100
    mass_balance_relgap[mass_balance_relgap.abs() < np.sqrt(eps)] = 0
    all_outputs["Mass balance gap [%]"] = mass_balance_relgap


    # Carbon balance [mol]:
    carbon_in = initials.Mx0 / constants.Mwx*1 + \
                initials.Mp0 / constants.Mwp*1 + \
                all_outputs["Σ glucose feed [g glucose]"] / constants.Mwglc*6
    carbon_out = all_outputs["Mglc [g glucose]"] * constants.Mwglc * 6 + \
                 all_outputs["Mx [g biomass X]"] / constants.Mwx * 1 + \
                 all_outputs["Mp [g protein p]"] / constants.Mwp * 1 + \
                 all_outputs["Σ CO2 [mol CO2]"] * 1
    carbon_gap = (carbon_in - carbon_out)/carbon_in * 100
    carbon_gap[carbon_gap.abs() < np.sqrt(eps)] = 0
    all_outputs["C mol balance gap [%]"] = carbon_gap


    # Nitrogen balance [mol]:
    nitrogen_in =  initials.Mx0 / constants.Mwx*0.2 + \
                   initials.Mp0 / constants.Mwp*0.3 + \
                   all_outputs["Σ NH3 feed [kg NH3]"] / constants.Mwnh3 * 1
    nitrogen_out = all_outputs["Mnh3 [g NH3]"] / constants.Mwnh3 * 1 + \
                   all_outputs["Mx [g biomass X]"] / constants.Mwx * 0.2 + \
                   all_outputs["Mp [g protein p]"] / constants.Mwp * 0.3
    nitrogen_gap = (nitrogen_in - nitrogen_out)/nitrogen_in * 100
    nitrogen_gap[nitrogen_gap.abs() < np.sqrt(eps)] = 0
    all_outputs["N mol balance gap [%]"] = nitrogen_gap
    
    return all_outputs


def diffeq(time, f, const, conditions):
    """
    Evaluation of differential equations.
    """
    
    # State variables:
    Mx =   f[3]                                                                # Biomass amount in broth [kg LDW]
    
    # Reaction rates:
    _, v, _ = kineq(time, f, const, conditions)    
    qglc = v[0]                                                                # Specific glucose uptake rate [g glc/g X/h]
    qnh3 = v[1]                                                                # Specific NH3 consumption rate [g nh3/g X/h]
    qo2  = v[2]                                                                # Specific O2 consumption rate [mol O2/g X/h]
    qx   = v[3]                                                                # Specific growth rate [1/h]
    qp   = v[4]                                                                # Specific protein formation rate [g p/g X/h]
    qco2 = v[5]                                                                # Specific CO2 production rate [mol CO2/g X/h]
    Fglc = v[6]                                                                # Glucose feed rate [kg glc/h]
    Fnh3 = v[7]                                                                # NH3 feed rate [kg nh3/h]

    # Differentials:
    dWBdt = (Fglc/const.Cglcf + Fnh3/const.Cnh3t + qo2 * Mx * const.Mwo2 - qco2 * Mx * const.Mwco2 - conditions.evap)/1000    # Change in broth weight (kg broth/h)
    dMglcdt = Fglc - qglc*Mx                                                   # Change in glucose (g glucose/h)
    dMnh3dt = Fnh3 - qnh3*Mx                                                   # Change in NH3 (g NH3/h)
    dMxdt = qx*Mx                                                              # Change in biomass (g X/h)
    dMpdt = qp*Mx                                                              # Change in protein (g p/h)
    dTglcdt = Fglc                                                             # Change in totalized glucose feed (g glucose/h)
    dTnh3dt = Fnh3                                                             # Change in totalized NH3 feed (kg NH3/h)
    dTo2dt = qo2*Mx                                                            # Change in totalized O2 (mol O2/h)
    dTco2dt = qco2*Mx                                                          # Change in totalized CO2 (mol CO2/h)
    dTh2odt = conditions.evap                                                  # Change in totalized evaporated water (g H2O/h)
    return [dWBdt, dMglcdt, dMnh3dt, dMxdt, dMpdt, dTglcdt, dTnh3dt, dTo2dt, dTco2dt, dTh2odt]


def kineq(time, f, const, conditions):
    """ Kinetic equations """
    
    # State variables:
    WB =   f[0]                                                                # Broth weight [kg]
    Mglc = f[1]                                                                # Glucose in broth [g glc]
    Mnh3 = f[2]                                                                # NH3 in broth [g nh3]
    Mx =   f[3]                                                                # Biomass in broth [g X]
    Mp =   f[4]                                                                # Protein in broth [g p]
    Tglc = f[5]                                                                # Totalized glc feed [g]
    assert(not np.isnan(WB))

    Cglc = Mglc/WB                                                             # Glucose concentration in broth [g/kg]
    Cnh3 = Mnh3/WB                                                             # NH3 concentration in broth [g/kg]
    Cx =   Mx/WB                                                               # Biomass concentration in broth [g/kg]
    Cp =   Mp/WB                                                               # Protein concentration in broth [g/kg]

    # Reaction rates:
    q1 = const.q1max * Cglc/(const.K1+Cglc) * Cnh3/(const.K2+Cnh3)             # Specific growth rate (Cmol X/Cmol X/h)
    q2 = const.K3 * q1 * (1-(const.K4/(1+np.exp(const.K5*(q1-const.K6)))))     # Specific protein formation rate (mol p/Cmol X/h)
    q4 = const.q4max * Cglc/(const.K7+Cglc)                                    # Maintenance (kJ/Cmol X/h)
    q3 = (const.G1*q1 + const.G2*q2 + q4)/const.G3                             # Glucose catabolism (mol glc/Cmol X/h)

    # (1) Growth reaction:
    # 0.175 C6H12O6 + 0.2 NH3 + 240 kJ -> 1 C1H1.8O0.5N0.2 + 0.05 CO2 + 0.45 H2O
    #
    # (2) Protein formation:
    # 0.171 C6H12O6 + 0.3 NH3 + 220 kJ -> 1 C1H1.6O0.3N0.3 + 0.025 CO2 + 0.675 H2O
    #
    # (3) Glucose catabolism:
    # 1 C6H12O6 + 6 O2 -> 6 CO2 + 6 H2O + 2843 kJ
    #
    # [qglc qnh3 qo2 qx  qp qco2]'=[-0.175 -0.17083333 -1 -0.2 -0.3 0 0 0 -6 1 0 0 0 1 0 0.05 0.025 6]*[q1 q2 q3]'
    q_vec =  np.array([[-0.175, -0.17083333, -1], # C6H12O6 consumed
                       [-0.2, -0.3, 0],           # NH3 consumed 
                       [0, 0, -6],                # O2 consumed
                       [1, 0, 0],                 # CH1.8O0.5N0.2 produced 
                       [0, 1, 0],                 # CH1.6O0.3N0.3 produced
                       [0.05, 0.025, 6]           # H20 produced
                      ])  @ np.array([q1, q2, q3])
                                                                    # (mol/cmol X/h)

    qglc = -q_vec[0]*const.Mwglc/const.Mwx                                     # Specific glc consumption [g glc/g X/h]
    qnh3 = -q_vec[1]*const.Mwnh3/const.Mwx                                     # Specific nh3 consumption [g nh3/g X/h]
    qo2  = -q_vec[2]/const.Mwx                                                 # Specific o2 consumption [mol o2/g X/h]
    qx   = q_vec[3]                                                            # Specific biomass growth [g X/g X/h]
    qp   = q_vec[4]*const.Mwp/const.Mwx                                        # Specific protein formation [g p/g X/h]
    qco2 = q_vec[5]/const.Mwx                                                  # Specific co2 production [mol co2/g X/h]

    OUR = qo2 * Cx * 1000                                                      # Oxygen uptake rate [mmol/kg/h]
    CPR = qco2 * Cx * 1000                                                     # Carbondioxide production rate [mmol/kg/h]

    # Glucose dosing:
    if (conditions.feed_glucose and CPR < 1 and time > 1) and not conditions.flagtime:
        # Criterion to end batch phase and switch to fed-batch
        # For the pure batch case 1: never use this section;
        # For the fed-batch case 2 only
        conditions.flagtime = time
    
    if conditions.flagtime and conditions.feed_glucose and WB <= 10:           # For case 2: stop when vessel reached 10 kg
        # Exponential feed rate (g glucose/h)
        if conditions.feed_profile.lower() == "exponential":
            Fglc = conditions.Fglc0 * np.exp(conditions.expF * (time - conditions.flagtime))  
            
        # Linear feed rate (g glucose/h) :   
        elif conditions.feed_profile.lower() == "linear":
            Fglc = conditions.Fglc0 + conditions.slopeF * (time - conditions.flagtime)      
    else:
        Fglc=0 

    # NH3 feeding:
    if conditions.flagtime and conditions.feed_glucose:
        Fnh3 = qnh3 * Mx                                                       # Feed rate ammonia (g ammonia/h)
    else:
        Fnh3=0 
            
    # Offgas:
    mv = 1.01e5 / 8.314 / (273 + conditions.temperature)                       # Mol per gas volume (mol/m3)
    ingasflw = conditions.airflow / 1000 * 60 * mv                             # Ingas flow (mol/h)
    offgasflw = ingasflw - qo2 * Mx + qco2 * Mx                                # Offgas flow (mol/h)
    yo2_out = (ingasflw * const.yo2_air - qo2*Mx) / offgasflw                  # O2 fraction offgas (mol/mol)
    yco2_out = (ingasflw * const.yco2_air + qco2*Mx) / offgasflw               # CO2 fraction offgas (mol/mol)
    offgasflw = offgasflw * 1000 / 60 / mv                                     # Offgas flow (Nl/min)

    c = [Cglc, Cnh3, Cx, Cp]                                                   # Concentration vector
    v = [qglc, qnh3, qo2, qx, qp, qco2, Fglc, Fnh3, OUR, CPR]                  # Rate vector
    y = [yo2_out, yco2_out, conditions.airflow, offgasflw]                     # Gas vector
    
    return c, v, y