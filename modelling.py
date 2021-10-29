from typing import Dict
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

eps = np.finfo(float).eps

#  =========== FIXED & CALCULATED CONSTANTS ================================


class Struct:
    pass


constants = Struct()
# Fixed constants:
constants.Mw = Struct()
constants.Mw.glc = 180  # Molweight glucose (g/mol)
constants.Mw.nh3 = 17  # Molweight NH3 (g/mol)
constants.Mw.x = (
    26  # Molweight biomass (g/mol), although 24.6 (g/mol) according formula: C1 H1.8 O0.5 N0.2
)
constants.Mw.p = (
    24  # Molweight protein (g/mol), although 22.6 (g/mol) according formula: C1 H1.6 O0.3 N0.3
)
constants.Mw.o2 = 32  # Molweight O2 (g/mol)
constants.Mw.co2 = 44  # Molweight CO2 (g/mol)
constants.Mw.ethd = 46.069  # Molweight Ethanol (g/mol)
constants.yo2_air = 0.21  # O2 fraction ingas (mol/mol)
constants.yco2_air = 0.0004  # CO2 fraction ingas (mol/mol)

# Feed solutions:
constants.Cglcf = 0.300  # Glucose concentration in feed solution (kg/kg)
constants.Cnh3t = 0.25  # NH3 concentration in titrant (kg/kg)

# Kinetics:
constants.q1max = 0.1  # Maximum specific growth rate (1/h)
constants.K1 = 0.1  # Affinity growth for glc (g/kg)
constants.K2 = 0.1  # Affinity growth for nh3 (g/kg)
constants.K3 = 0.1  # Ratio protein formation rate to growth (-)
constants.K4 = 2  # Decline of protein formation towards max growth (-)
constants.K5 = -100  # Onset decline of protein formation rate (kg/g)
constants.K6 = 0.1  # Onset decline of protein formation rate (g/kg)
constants.q4max = 4.5  # Maximum maintenance (kJ/Cmol X/h)
constants.K7 = 0.01  # Affinity maintenance for glc (g/kg)
constants.K8 = 0.01e-3  # Affinity growth and maintenance for O2 (mol/kg)
constants.G1 = 240  # Energy consumption for growth (kJ/cmol X)
constants.G2 = 220  # Energy consumption for protein formation (kJ/Cmol p)
constants.G3 = 2843  # Energy generation by glc catabolism (kJ/mol glc)

# Mass transfer
constants.kla_o2 = 250  # Mass transfer coefficient for O2 (h-1)
constants.kla_co2 = constants.kla_o2 * 1.91 / 2.42  # Mass transfer coefficient for CO2 (h-1)


def call_fermentation_model(settings):
    """
    Gets all the settings, initial conditions and constants, and does the ODE integration.
    Returns just the bare integration results.
    """
    print("Running simulation ... ")
    # Defaults are "Full batch operation": no in or out flow of anything,
    # .             no titrant, no glucose, no product withdrawal.
    defaults = {
        "Glucose": "No addition",  # Indicates to never feed glucose
        "Cglc0": 50,  # Initial glucose concentration (g/kg) in the vessel
        "Cnh30": 3.0,  # Initial NH3 concentration (g/kg)
        "Cx0": 1,  # Initial biomass concentration (g/kg) [inoculation]
        "evap": 0,  # Evaporation rate (g/h)
        "WB0": 10,  # Initial broth weight (kg) in the vessel
        "Duration": 66,  # Duration of fermentation (h)
    }

    # But then update the parts of the settings which override this
    defaults.update(settings)
    conditions, initials = get_conditions_initials(settings)

    # Then do the simulation
    out = solve_ivp(
        diffeq,
        t_span=(0, conditions.Tend),
        y0=[
            initials.WB0,
            initials.Mglc0,
            initials.Mnh30,
            initials.Mx0,
            initials.Mp0,
            initials.Tglc0,
            initials.Tnh30,
            initials.To20,
            initials.Tco20,
            initials.Th2o0,
            initials.Mo20,  # These 4 are new for the mass transfer case
            initials.Mco20,
            initials.Mo2g0,
            initials.Mco2g0,
        ],
        t_eval=np.arange(start=0, step=conditions.Tint, stop=conditions.Tend),
        args=(constants, conditions),
        method="RK23",
    )
    print("Simulation completed. Generating plots ...")
    return out


def condition_dependent_constants(temperature):
    return dict(
        mv=101325 / 8.31446261815324 / (273.15 + temperature),  # Mol per gas volume (mol/m3)
        Ho2=0.032
        * np.exp(
            1700 * ((1 / (273 + temperature)) - (1 / (273 + 25)))
        ),  # Henri coefficient O2 (mol aqueous / mol gas)
        Hco2=0.83
        * np.exp(
            2400 * ((1 / (273 + temperature)) - (1 / (273 + 25)))
        ),  # Henri coefficient CO2 (mol aqueous / mol gas)
    )


def get_conditions_initials(settings: Dict):
    class Struct:
        pass

    conditions = Struct()

    conditions.Tend = settings["Duration"]
    conditions.Tint = 0.25  # Time resolution of simulation (h)
    conditions.airflow = 10  # Airflow rate (NL/min)
    conditions.Fglc0 = 3.0  # Starting glucose feed rate (g/h) when glucose is required
    conditions.expF = 0.07  # Exponential increase of feed rate (1/h)
    conditions.slopeF = 1.0  # Slope of feed rate (g/h/h), if feeding linearly.
    conditions.feed_profile = (
        "linear" if settings["Glucose"] == "Linear: if CPR≤1 and WB≤10" else "exponential"
    )
    conditions.temperature = 25  # Fermentation T (C)
    conditions.evap = settings["Evaporation"]
    conditions.feed_glucose = not (settings["Glucose"] == "No addition")
    conditions.flagtime = 0
    conditions.Volume = 15  # Gross volume of fermenter (L)
    conditions.mass_transfer = True if settings["Mass transfer"] == "Gas ⇆ Liquid" else False
    assert conditions.feed_profile in ["linear", "exponential"]

    initials = Struct()
    initials.Cglc0 = settings["C glc t0"]  # Initial glucose concentration (g/kg)
    initials.Cnh30 = settings["C NH3 t0"]  # Initial NH3 concentration (g/kg)
    initials.Cx0 = settings["C X t0"]  # Initial biomass concentration (g/kg)
    initials.WB0 = settings["WB t0"]  # Initial broth weight (kg)
    initials.Cp0 = 0  # Initial protein concentration (g/kg)
    initials.Mglc0 = initials.Cglc0 * initials.WB0  # Initial glucose weight (g)
    initials.Mnh30 = initials.Cnh30 * initials.WB0  # Initial NH3 weight (g)
    initials.Mx0 = initials.Cx0 * initials.WB0  # Initial biomass weight (g)
    initials.Mp0 = initials.Cp0 * initials.WB0  # Initial protein weight (g)
    initials.Tglc0 = initials.Mglc0  # Totalized glucose dosed (g)
    initials.Tnh30 = initials.Mnh30  # Totalized NH3 dosed (g)
    initials.To20 = 0  # Totalized O2 consumed (mol)
    initials.Tco20 = 0  # Totalized CO2 produced (mol)
    initials.Th2o0 = 0  # Totalized H2O evaporated (g)

    # Mass transfer: 4 extra term: Mo20, Mco20, Mo2g0, Mco2g0
    cdc = condition_dependent_constants(conditions.temperature)
    initials.Mo20 = (
        cdc["mv"] * constants.yo2_air * cdc["Ho2"] / 1000 * initials.WB0
    )  # Initial dissolved O2 moles (mol)
    initials.Mco20 = (
        cdc["mv"] * constants.yco2_air * cdc["Hco2"] / 1000 * initials.WB0
    )  # Initial dissolved cO2 moles (mol)
    initials.Mg0 = (
        (conditions.Volume - initials.WB0) / 1000 * cdc["mv"]
    )  # Initial amount of gas in fermenter (mol)
    initials.Mo2g0 = initials.Mg0 * constants.yo2_air  # Initial O2 moles in gas phase (mol)
    initials.Mco2g0 = initials.Mg0 * constants.yco2_air  # Initial CO2 moles in gas phase (mol)

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
        "Σ NH3 feed [g NH3]",
        "Σ O2 [mol O2]",
        "Σ CO2 [mol CO2]",
        "Σ evap H2O [g H2O]",
        "O2 dissolved [mol]",
        "CO2 dissolved [mol]",
        "O2 in gas phase [mol]",
        "CO2 in gas phass [mol]",
    ]
    outputs = pd.DataFrame(profiles.y).T
    outputs.columns = columns
    outputs["Time [hours]"] = profiles.t

    extra_columns = []
    for _, values in outputs.iterrows():
        addition = {}
        c, v, y = kineq(
            values["Time [hours]"], values.values, const=constants, conditions=conditions
        )

        (
            addition["Cglc [g/kg]"],
            addition["Cnh3 [g/kg]"],
            addition["Cx [g/kg]"],
            addition["Cp g/kg]"],
        ) = c
        (
            addition["Qglc [g/g X/h]"],
            addition["Qnh3 [g/g X/h]"],
            addition["Qo2 [mol/g X/h]"],
            addition["Qx [1/h]"],
            addition["Qp [g/g X/h]"],
            addition["Qco2 [mol/g X/h]"],
            _,
            _,
            addition["OUR [mmol/kg/h]"],
            addition["CPR [mmol/kg/h]"],
        ) = v
        (
            addition["yO2 [mol/mol]"],
            addition["yCO2 [mol/mol]"],
            addition["Airflow [NL/min]"],
            addition["Offgas flow [NL/min]"],
        ) = y
        addition["Time [hours]"] = values["Time [hours]"]
        assert (
            len([v for v in addition.values() if np.isnan(v)]) == 0
        ), f"There are missing values at {addition['Time [hours]']} hours"
        extra_columns.append(addition)

    # Merge datasets, based on the time values
    all_outputs = outputs.set_index("Time [hours]").join(
        pd.DataFrame(extra_columns).set_index("Time [hours]")
    )
    all_outputs["Fglc [g/h]"] = all_outputs["Σ glucose feed [g glucose]"].diff()
    all_outputs["Fnh3 [g/h]"] = all_outputs["Σ NH3 feed [g NH3]"].diff()

    # But then reset the index, to allow further calculations
    all_outputs = all_outputs.reset_index()
    all_outputs["Rglc [g/h]"] = all_outputs["Qglc [g/g X/h]"] * all_outputs["Mx [g biomass X]"]
    all_outputs["Rnh3 [g/h]"] = all_outputs["Qnh3 [g/g X/h]"] * all_outputs["Mx [g biomass X]"]
    all_outputs["Rx [g/h]"] = all_outputs["Qx [1/h]"] * all_outputs["Mx [g biomass X]"]
    all_outputs["Rp [g/h]"] = all_outputs["Qp [g/g X/h]"] * all_outputs["Mx [g biomass X]"]
    all_outputs["RQ [-]"] = all_outputs["CPR [mmol/kg/h]"] / all_outputs["OUR [mmol/kg/h]"]
    all_outputs["Yxs [g/g]"] = (all_outputs["Mx [g biomass X]"] - initials.Mx0) / (
        all_outputs["Σ glucose feed [g glucose]"] - all_outputs["Mglc [g glucose]"]
    )
    all_outputs["Yps [g/g]"] = (all_outputs["Mp [g protein p]"] - initials.Mp0) / (
        all_outputs["Σ glucose feed [g glucose]"] - all_outputs["Mglc [g glucose]"]
    )
    all_outputs["Yos [g/g]"] = (
        (all_outputs["Σ O2 [mol O2]"])
        / (all_outputs["Σ glucose feed [g glucose]"] - all_outputs["Mglc [g glucose]"])
        * 1000
    )
    all_outputs["Prod [g/kg/h]"] = (
        all_outputs["Mp [g protein p]"]
        / all_outputs["Broth weight [kg broth]"]
        / all_outputs["Time [hours]"]
    )

    # Gas-phase: 4 extra outputs from mass transfer
    cdc = condition_dependent_constants(conditions.temperature)
    all_outputs["C_{O2}*g [mmol/kg]"] = all_outputs["yO2 [mol/mol]"] * cdc["mv"] * cdc["Ho2"]
    all_outputs["C_{CO2}*g [mmol/kg]"] = all_outputs["yCO2 [mol/mol]"] * cdc["mv"] * cdc["Hco2"]
    all_outputs["C_{O2} [mmol/kg]"] = (
        all_outputs["O2 dissolved [mol]"] * all_outputs["Broth weight [kg broth]"] * 1000
    )
    all_outputs["C_{CO2} [mmol/kg]"] = (
        all_outputs["CO2 dissolved [mol]"] * all_outputs["Broth weight [kg broth]"] * 1000
    )

    # Force small values to zero
    for colname, column in all_outputs.iteritems():
        all_outputs.loc[column.abs() < np.sqrt(eps), colname] = 0

    # Mass balance [grams]:
    mass_in = (
        initials.WB0 * 1000
        + (all_outputs["Σ glucose feed [g glucose]"] - initials.Mglc0) / constants.Cglcf
        + (all_outputs["Σ NH3 feed [g NH3]"] - initials.Mnh30) / constants.Cnh3t
        + (all_outputs["Σ O2 [mol O2]"] * constants.Mw.o2)
    )
    mass_out = (
        all_outputs["Broth weight [kg broth]"] * 1000
        + all_outputs["Σ CO2 [mol CO2]"] * constants.Mw.co2
        + all_outputs["Σ evap H2O [g H2O]"]
    )
    mass_balance_relgap = (mass_in - mass_out) / mass_in * 100
    mass_balance_relgap[mass_balance_relgap.abs() < np.sqrt(eps)] = 0
    all_outputs["Mass balance gap [%]"] = mass_balance_relgap

    # Carbon balance [mol]:
    carbon_in = (
        initials.Mx0 / constants.Mw.x * 1
        + initials.Mp0 / constants.Mw.p * 1
        + all_outputs["Σ glucose feed [g glucose]"] / constants.Mw.glc * 6
    )
    carbon_out = (
        all_outputs["Mglc [g glucose]"] * constants.Mw.glc * 6
        + all_outputs["Mx [g biomass X]"] / constants.Mw.x * 1
        + all_outputs["Mp [g protein p]"] / constants.Mw.p * 1
        + all_outputs["Σ CO2 [mol CO2]"] * 1
    )
    carbon_gap = (carbon_in - carbon_out) / carbon_in * 100
    carbon_gap[carbon_gap.abs() < np.sqrt(eps)] = 0
    all_outputs["C mol balance gap [%]"] = carbon_gap

    # Nitrogen balance [mol]:
    nitrogen_in = (
        initials.Mx0 / constants.Mw.x * 0.2
        + initials.Mp0 / constants.Mw.p * 0.3
        + all_outputs["Σ NH3 feed [g NH3]"] / constants.Mw.nh3 * 1
    )
    nitrogen_out = (
        all_outputs["Mnh3 [g NH3]"] / constants.Mw.nh3 * 1
        + all_outputs["Mx [g biomass X]"] / constants.Mw.x * 0.2
        + all_outputs["Mp [g protein p]"] / constants.Mw.p * 0.3
    )
    nitrogen_gap = (nitrogen_in - nitrogen_out) / nitrogen_in * 100
    nitrogen_gap[nitrogen_gap.abs() < np.sqrt(eps)] = 0
    all_outputs["N mol balance gap [%]"] = nitrogen_gap

    return all_outputs


def diffeq(time, f, const, conditions):
    """
    Evaluation of differential equations.
    """
    cdc = condition_dependent_constants(conditions.temperature)

    # State variables:
    WB = f[0]  # Broth weight (kg)
    Mx = f[3]  # Biomass amount in broth [kg LDW]
    Mo2 = f[10]  # Moles of O2 in broth (mol)
    Mco2 = f[11]  # Moles of CO2 in broth (mol)
    Co2 = Mo2 / WB  # Concentration of O2 in broth (mol/kg)
    Cco2 = Mco2 / WB  # Concentration of CO2 in broth (mol/kg)
    # Vg = conditions.Volume - WB  # Gas volume in fermenter (l). NB: broth density assumed 1 kg/k!
    Mg = (conditions.Volume - WB) / 1000 * cdc["mv"]  # Amount of gas in fermenter (mol)
    Mo2g = f[12]  # O2 moles in gas phase (mol)
    Mco2g = f[13]  # CO2 moles in gas phase (mol)
    yo2 = Mo2g / Mg  # Mole fraction of O2 in gas phase (mol/mol)
    yco2 = Mco2g / Mg  # Mole fraction of CO2 in gas phase (mol/mol)
    Co2_ = cdc["mv"] * yo2 * cdc["Ho2"] / 1000  # O2 concentration at bubble surface (mol/kg)
    Cco2_ = cdc["mv"] * yco2 * cdc["Hco2"] / 1000  # CO2 concentration at bubble surface (mol/kg)

    # Reaction rates:
    _, v, _ = kineq(time, f, const, conditions)
    qglc = v[0]  # Specific glucose uptake rate [g glc/g X/h]
    qnh3 = v[1]  # Specific NH3 consumption rate [g nh3/g X/h]
    qo2 = v[2]  # Specific O2 consumption rate [mol O2/g X/h]
    qx = v[3]  # Specific growth rate [1/h]
    qp = v[4]  # Specific protein formation rate [g p/g X/h]
    qco2 = v[5]  # Specific CO2 production rate [mol CO2/g X/h]
    Fglc = v[6]  # Glucose feed rate [kg glc/h]
    Fnh3 = v[7]  # NH3 feed rate [kg nh3/h]

    # Differentials:
    dWBdt = (
        Fglc / const.Cglcf
        + Fnh3 / const.Cnh3t
        + qo2 * Mx * const.Mw.o2
        - qco2 * Mx * const.Mw.co2
        - conditions.evap
    ) / 1000  # Change in broth weight (kg broth/h)
    dMglcdt = Fglc - qglc * Mx  # Change in glucose (g glucose/h)
    dMnh3dt = Fnh3 - qnh3 * Mx  # Change in NH3 (g NH3/h)
    dMxdt = qx * Mx  # Change in biomass (g X/h)
    dMpdt = qp * Mx  # Change in protein (g p/h)
    dTglcdt = Fglc  # Change in totalized glucose feed (g glucose/h)
    dTnh3dt = Fnh3  # Change in totalized NH3 feed (kg NH3/h)
    dTh2odt = conditions.evap  # Change in totalized evaporated water (g H2O/h)

    # Differentials, continued:
    if conditions.mass_transfer:
        o2_tr = WB * const.kla_o2 * (Co2_ - Co2)  # O2 transfer G->L (mol/h)
        co2_tr = WB * const.kla_co2 * (Cco2 - Cco2_)  # CO2 transfer L->G (mol/h)
        ingasflw = conditions.airflow / 1000 * 60 * cdc["mv"]  # Ingas flow (mol/h)
        offgasflw = (
            ingasflw - o2_tr + co2_tr + dWBdt / 1000 * cdc["mv"]
        )  # Offgas flow (mol/h). NB: last term represents gas replaced by broth
        dTo2dt = o2_tr  # Change in totalized O2 (mol O2/h)
        dTco2dt = co2_tr  # Change in totalized CO2 (mol CO2/h)

        dMo2dt = -qo2 * Mx + o2_tr
        # Change in dissolved O2 (mol O2/h)
        dMco2dt = qco2 * Mx - co2_tr
        # Change in dissolved CO2 (mol CO2/h)
        dMo2gdt = (
            -o2_tr + ingasflw * const.yo2_air - offgasflw * yo2
        )  # Change in O2 in gas phase (mol O2/h)
        dMco2gdt = (
            co2_tr + ingasflw * const.yco2_air - offgasflw * yco2
        )  # Change in CO2 in gas phase (mol CO2/h)
    else:
        dTo2dt = qo2 * Mx  # Change in totalized O2 (mol O2/h)
        dTco2dt = qco2 * Mx  # Change in totalized CO2 (mol CO2/h)
        dMo2dt = 0.0
        dMco2dt = 0.0
        dMo2gdt = 0.0
        dMco2gdt = 0.0

    return [
        dWBdt,
        dMglcdt,
        dMnh3dt,
        dMxdt,
        dMpdt,
        dTglcdt,
        dTnh3dt,
        dTo2dt,
        dTco2dt,
        dTh2odt,
        dMo2dt,
        dMco2dt,
        dMo2gdt,
        dMco2gdt,
    ]


def kineq(time, f, const, conditions):
    """Kinetic equations"""
    cdc = condition_dependent_constants(conditions.temperature)
    # State variables:
    WB = f[0]  # Broth weight [kg]
    Mglc = f[1]  # Glucose in broth [g glc]
    Mnh3 = f[2]  # NH3 in broth [g nh3]
    Mx = f[3]  # Biomass in broth [g X]
    Mp = f[4]  # Protein in broth [g p]
    # Tglc = f[5]  # Totalized glc feed [g]
    # Mass transfer state variables
    Mo2 = f[10]  # Moles of O2 in broth (mol)
    Mco2 = f[11]  # Moles of CO2 in broth (mol)
    Mo2g = f[12]  # O2 moles in gas phase (mol)
    Mco2g = f[13]  # CO2 moles in gas phase (mol)

    assert not np.isnan(WB), "Broth weight cannot be NaN"
    assert WB > 0, "Broth weight must be positive"

    Cglc = Mglc / WB  # Glucose concentration in broth [g/kg]
    Cnh3 = Mnh3 / WB  # NH3 concentration in broth [g/kg]
    Cx = Mx / WB  # Biomass concentration in broth [g/kg]
    Cp = Mp / WB  # Protein concentration in broth [g/kg]
    Co2 = Mo2 / WB  # Concentration of O2 in broth (mol/kg)
    Cco2 = Mco2 / WB  # Concentration of CO2 in broth (mol/kg)
    # Vg = conditions.Volume - WB  # Gas volume in fermenter (l). NB: broth density assumed 1 kg/k!
    Mg = (conditions.Volume - WB) / 1000 * cdc["mv"]  # Amount of gas in fermenter (mol)
    yo2 = Mo2g / Mg  # Mole fraction of O2 in gas phase (mol/mol)
    yco2 = Mco2g / Mg  # Mole fraction of CO2 in gas phase (mol/mol)
    Co2_ = cdc["mv"] * yo2 * cdc["Ho2"] / 1000  # O2 concentration at bubble surface (mol/kg)
    Cco2_ = cdc["mv"] * yco2 * cdc["Hco2"] / 1000  # CO2 concentration at bubble surface (mol/kg)

    # Reaction rates:
    # Specific growth rate (Cmol X/Cmol X/h)
    o2_rate_multiplier = (
        Co2 / (const.K8 + Co2) if conditions.mass_transfer else 1.0
    )  # Extra term if mass transfer
    q1 = const.q1max * Cglc / (const.K1 + Cglc) * Cnh3 / (const.K2 + Cnh3) * o2_rate_multiplier
    q2 = (
        const.K3 * q1 * (1 - (const.K4 / (1 + np.exp(const.K5 * (q1 - const.K6)))))
    )  # Specific protein formation rate (mol p/Cmol X/h)
    q4 = const.q4max * Cglc / (const.K7 + Cglc) * o2_rate_multiplier  # Maintenance (kJ/Cmol X/h)
    q3 = (const.G1 * q1 + const.G2 * q2 + q4) / const.G3  # Glucose catabolism (mol glc/Cmol X/h)

    # (1) Growth reaction:
    # 0.175 C6H12O6 + 0.2 NH3 + 240 kJ -> 1 C1H1.8O0.5N0.2 + 0.05 CO2 + 0.45 H2O
    #
    # (2) Protein formation:
    # 0.171 C6H12O6 + 0.3 NH3 + 220 kJ -> 1 C1H1.6O0.3N0.3 + 0.025 CO2 + 0.675 H2O
    #
    # (3) Glucose catabolism:
    # 1 C6H12O6 + 6 O2 -> 6 CO2 + 6 H2O + 2843 kJ
    #
    # [qglc qnh3 qo2 qx  qp qco2]'= \
    #         [-0.175 -0.17083333 -1, -0.2 -0.3 0, 0 0 -6, 1 0 0, 0 1 0, 0.05 0.025 6]*[q1 q2 q3]'
    q_vec = (
        np.array(
            [
                [-0.175, -0.17083333, -1],  # C6H12O6 consumed
                [-0.2, -0.3, 0],  # NH3 consumed
                [0, 0, -6],  # O2 consumed
                [1, 0, 0],  # CH1.8O0.5N0.2 produced
                [0, 1, 0],  # CH1.6O0.3N0.3 produced
                [0.05, 0.025, 6],  # H20 produced
            ]
        )
        @ np.array([q1, q2, q3])
    )
    # (mol/cmol X/h)
    qglc = -q_vec[0] * const.Mw.glc / const.Mw.x  # Specific glc consumption [g glc/g X/h]
    qnh3 = -q_vec[1] * const.Mw.nh3 / const.Mw.x  # Specific nh3 consumption [g nh3/g X/h]
    qo2 = -q_vec[2] / const.Mw.x  # Specific o2 consumption [mol o2/g X/h]
    qx = q_vec[3]  # Specific biomass growth [g X/g X/h]
    qp = q_vec[4] * const.Mw.p / const.Mw.x  # Specific protein formation [g p/g X/h]
    qco2 = q_vec[5] / const.Mw.x  # Specific co2 production [mol co2/g X/h]
    OUR = qo2 * Cx * 1000  # Oxygen uptake rate [mmol/kg/h]
    CPR = qco2 * Cx * 1000  # Carbondioxide production rate [mmol/kg/h]

    # Glucose dosing (g glucose/h)
    Fglc = 0.0
    if (conditions.feed_glucose and CPR < 1 and time > 1) and not conditions.flagtime:
        # Criterion to end batch phase and switch to fed-batch
        # For the pure batch case 1: never use this section;
        # For the fed-batch case 2 only

        conditions.flagtime = time

    if (
        (conditions.flagtime > 0) and conditions.feed_glucose and WB <= 10
    ):  # For case 2: stop when vessel reached 10 kg
        # Exponential feed rate (g glucose/h)
        if conditions.feed_profile.lower() == "exponential":
            Fglc = conditions.Fglc0 * np.exp(conditions.expF * (time - conditions.flagtime))

        # Linear feed rate (g glucose/h) :
        elif conditions.feed_profile.lower() == "linear":
            Fglc = conditions.Fglc0 + conditions.slopeF * (time - conditions.flagtime)

    # Feed rate ammonia (g ammonia/h)
    Fnh3 = 0.0
    if (conditions.flagtime > 0) and conditions.feed_glucose:
        Fnh3 = qnh3 * Mx

    # Offgas:
    o2_tr = WB * constants.kla_o2 * (Co2_ - Co2)  # O2 transfer G->L (mol/kg/h)
    co2_tr = WB * constants.kla_co2 * (Cco2 - Cco2_)  # CO2 transfer L->G (mol/kg/h)
    ingasflw = conditions.airflow / 1000 * 60 * cdc["mv"]  # Ingas flow (mol/h)
    dWBdt = (
        Fglc / const.Cglcf
        + Fnh3 / const.Cnh3t
        + qo2 * Mx * const.Mw.o2
        - qco2 * Mx * const.Mw.co2
        - conditions.evap
    ) / 1000  # Change in broth weight (kg broth/h)

    if conditions.mass_transfer:
        offgasflw = (
            ingasflw - o2_tr + co2_tr + dWBdt / 1000 * cdc["mv"]
        )  # Offgas flow (mol/h). NB: last term represents gas replaced by broth
        yo2_out, yco2_out = yo2, yco2
    else:
        offgasflw = ingasflw - qo2 * Mx + qco2 * Mx  # Offgas flow (mol/h)
        yo2_out = (ingasflw * const.yo2_air - qo2 * Mx) / offgasflw  # O2 fraction offgas (mol/mol)
        yco2_out = (
            ingasflw * const.yco2_air + qco2 * Mx
        ) / offgasflw  # CO2 fraction offgas (mol/mol)

    offgasflw_nl = offgasflw * 1000 / 60 / cdc["mv"]  # Offgas flow (Nl/min)
    c = [Cglc, Cnh3, Cx, Cp]  # Concentration vector
    v = [qglc, qnh3, qo2, qx, qp, qco2, Fglc, Fnh3, OUR, CPR]  # Rate vector
    y = [yo2_out, yco2_out, conditions.airflow, offgasflw_nl]  # Gas vector
    return c, v, y
