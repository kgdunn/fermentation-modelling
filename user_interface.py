from process_improve.batch.plotting import plot_multitags

# Functions from this app/repo
from modelling import call_fermentation_model, complete_balance_calculations
from ipywidgets import Output, Layout, Box, VBox, HBox
from ipysheet import sheet, column, row
import ipywidgets as widgets

max_scenarios = 5
df_dict = {}
colour_order = [
    "rgb(31, 119, 180)",  # muted blue
    "rgb(255, 127, 14)",  # safety orange
    "rgb(148, 103, 189)",  # muted purple
    "rgb(1, 99, 58)",  # green
    "rgb(179, 66, 66)",  # redish
]
assert len(colour_order) == max_scenarios
search_space = {
    "C glc t0": dict(
        style=widgets.FloatSlider,
        min=1,
        max=50,
        value=50,
        step=1,
        readout_format=".0f",
        units="g/kg",
        readout=True,
    ),
    "C NH3 t0": dict(
        style=widgets.FloatSlider,
        min=2.0,
        max=4.0,
        value=3.0,
        step=0.2,
        readout_format=".1f",
        units="g/kg",
        readout=True,
    ),
    "C X t0": dict(
        style=widgets.FloatSlider,
        min=1.0,
        max=4.0,
        value=1.0,
        step=0.2,
        readout_format=".1f",
        units="g/kg",
        readout=True,
    ),
    "Evaporation": dict(
        style=widgets.FloatSlider,
        min=0,
        max=20.0,
        value=0.0,
        step=1.0,
        readout_format=".1f",
        units="g/hr",
        readout=True,
    ),
    "WB t0": dict(
        style=widgets.FloatSlider,
        min=1,
        max=15.0,
        value=10.0,
        step=1.0,
        readout_format=".0f",
        units="kg",
        readout=True,
    ),
    "Duration": dict(
        style=widgets.FloatSlider,
        min=40,
        max=80,
        value=66,
        step=2,
        readout_format=".0f",
        units="hours",
        readout=True,
    ),
    "Glucose": dict(
        style=widgets.RadioButtons,
        options=["No addition", "Linear: if CPR≤1 and WB≤10", "Exponential: if CPR≤1 and WB≤10"],
        units="",
    ),
    "Mass transfer": dict(
        style=widgets.RadioButtons,
        options=["None", "Gas ⇆ Liquid"],
        units="",
    ),
}
plot_groups = {
    "Masses": [
        "Broth weight [kg broth]",
        "Mglc [g glucose]",
        "Mnh3 [g NH3]",
        "Mx [g biomass X]",
        "Mp [g protein p]",
    ],
    "Totalizers": [
        "Σ glucose feed [g glucose]",
        "Σ NH3 feed [g NH3]",
        "Σ O2 [mol O2]",
        "Σ CO2 [mol CO2]",
        "Σ evap H2O [g H2O]",
    ],
    "Concentrations": [
        "Cglc [g/kg]",
        "Cnh3 [g/kg]",
        "Cx [g/kg]",
        "Cp g/kg]",
    ],
    "Specific Q rates": [
        "Qglc [g/g X/h]",
        "Qnh3 [g/g X/h]",
        "Qo2 [mol/g X/h]",
        "Qx [1/h]",
        "Qp [g/g X/h]",
        "Qco2 [mol/g X/h]",
    ],
    "G⇆L mass transfer": [
        "yO2 [mol/mol]",
        "yCO2 [mol/mol]",
        "Airflow [NL/min]",
        "Offgas flow [NL/min]",
        "C_{O2}*g [mmol/kg]",
        "C_{CO2}*g [mmol/kg]",
        "C_{O2} [mmol/kg]",
        "C_{CO2} [mmol/kg]",
    ],
    "Inlet flows": [
        "Fglc [g/h]",
        "Fnh3 [g/h]",
    ],
    "Rates/uptakes": [
        "OUR [mmol/kg/h]",
        "CPR [mmol/kg/h]",
        "RQ [-]",
        "Rglc [g/h]",
        "Rnh3 [g/h]",
        "Rx [g/h]",
        "Rp [g/h]",
    ],
    "KPIs": [
        "Yxs [g/g]",
        "Yps [g/g]",
        "Yos [g/g]",
        "Prod [g/kg/h]",
    ],
    "Balances": ["Mass balance gap [%]", "C mol balance gap [%]", "N mol balance gap [%]"],
}
search_space_names = sorted(list(set(search_space.keys())))
input_values = []
for key, val in search_space.items():
    values = val.copy()
    input_values.append(
        values.pop("style")(
            description=key,
            disabled=False,
            continuous_update=True,
            orientation="horizontal",
            tooltip=val["units"],
            layout=widgets.Layout(width="400px"),
            **values,
        )
    )

sim_settings = []
for idx in range(max_scenarios):
    subset = [widgets.Text(layout=widgets.Layout(width="100px")) for _ in search_space_names]
    sim_settings.append(subset)


def generate_plots(df_dict, n_frames=100):
    # Lines become thinner as the scenarios fade.
    highlight = {
        f'{{"width": {max_scenarios - len(df_dict) + idx + 1}, "color": "{colour_order[idx]}"}}': [
            val
        ]
        for idx, val in enumerate(df_dict.keys())
    }

    tag_list = []
    for pgc in plot_group_checks:
        if pgc.value:
            tag_list.extend(plot_groups[pgc.description])

    return plot_multitags(
        df_dict=df_dict,
        time_column="Time [hours]",
        tag_list=tag_list,
        batches_to_highlight=highlight,
        settings=dict(
            nrows=5,
            show_legend=False,
            html_image_height=1000,
            html_aspect_ratio_w_over_h=1.6,
            animate=False,
        ),
    )


plot_group_checks = [widgets.Checkbox(description=key, value=True) for key in plot_groups.keys()]


def create_user_interface(max_scenarios=5):
    """Creates the user interface"""
    headers = ["Input value", "Units"]
    headers.extend([f"Scenario {i+1}" for i in range(max_scenarios)])
    widths = [500, 100]
    widths.extend([200 for i in range(max_scenarios)])
    s1 = sheet(
        rows=len(search_space.keys()) + 1, columns=7, column_headers=headers, column_width=widths
    )

    column(0, input_values, read_only=True, type="widget")
    column(
        1,
        [val["units"] for _, val in search_space.items()],
        read_only=True,
    )

    for idx in range(max_scenarios):
        column(
            idx + 2,
            sim_settings[idx],
            read_only=True,
        )

    startoff = [
        widgets.HTML(
            value="<span style='float:right'><b>Colour</b></span>",
            layout=widgets.Layout(height="3em"),
        ),
        widgets.HTML(value="", placeholder="", description=""),
    ]
    startoff.extend(
        [
            widgets.HTML(
                value=f"<span style='color: {colour_order[idx]}'><b>Scenario {idx+1}<b></span>"
            )
            for idx in range(max_scenarios)
        ]
    )
    row(len(search_space), startoff)
    solve_button = widgets.Button(description="Simulate →")
    solve_button.on_click(run_simulation)

    last_row = [solve_button]
    last_row.extend(plot_group_checks)
    box_layout = Layout(
        display="inline-flex", flex_flow="column", align_items="flex-start", width="100%"
    )
    box_auto = Box(
        children=[
            s1,
            HBox(
                last_row,
                layout=Layout(
                    display="inline-flex",
                    flex_flow="flex-wrap wrap",
                    align_items="flex-start",
                    align_content="flex-start",
                    width="2000px",
                ),
            ),
        ],
        layout=box_layout,
    )
    display(VBox([box_auto, text_out]))


text_out = Output()


@text_out.capture(clear_output=True, wait=True)
def run_simulation(x_search=None):
    """Runs the ODE integrator simulation, calculates extra columns and plots all the results."""
    user_input_dict = {val.description: val.value for val in input_values}
    if len(df_dict) == max_scenarios:
        print("Maximum number of scenarios reached. Stopping to avoid overwriting results.")
        display(generate_plots(df_dict, n_frames=user_input_dict["Duration"] / 2))

    outputs = call_fermentation_model(settings=user_input_dict)
    all_outputs = complete_balance_calculations(outputs, settings=user_input_dict)

    df_dict[f"Scenario {len(df_dict)+1}"] = all_outputs
    vector = sim_settings[len(df_dict) - 1]
    for idx, val in enumerate(user_input_dict.values()):
        vector[idx].value = f"{val}"

    display(generate_plots(df_dict, n_frames=user_input_dict["Duration"] / 2))
