import numpy as np
import pandas as pd

from process_improve.batch.plotting import plot_multitags

# Functions from this app/repo
from modelling import call_fermentation_model, complete_balance_calculations
from ipywidgets import Output, Layout, Box, VBox, HBox
from ipysheet import sheet, column, to_dataframe, row
import ipywidgets as widgets

max_scenarios = 5
df_dict={}
colour_order = [        
    "rgb(31, 119, 180)",   # muted blue
    "rgb(255, 127, 14)",   # safety orange        
    "rgb(148, 103, 189)",  # muted purple
    "rgb(1, 99, 58)",      # green
    "rgb(179, 66, 66)",    # redish
]
assert len(colour_order) == max_scenarios
search_space={
    "Add glucose?": dict(
        min=0,
        max=1.0,
        value=0,
        step=1,
        readout_format=".0f",
        units="No/Yes",
    ),
    "C glc t0": dict(
        min=20,
        max=50,
        value=50,
        step=10,
        readout_format=".0f",
        units="g/kg",
    ),
    "C NH3 t0": dict(
        min=2.0,
        max=4.0,
        value=3.0,
        step=0.2,
        readout_format=".1f",
        units="g/kg",
    ),
    "C X t0": dict(
        min=1.0,
        max=4.0,
        value=1.0,
        step=0.2,
        readout_format=".1f",
        units="g/kg",
    ),
    "Evaporation": dict(
        min=0,
        max=4.0,
        value=0.0,
        step=1.,
        readout_format=".1f",
        units="g/hr",
    ),
    "WB t0": dict(
        min=1,
        max=15.0,
        value=10.0,
        step=1.0,
        readout_format=".0f",
        units="kg",
    ),
    "Duration": dict(
        min=40,
        max=80,
        value=66,
        step=2,
        readout_format=".0f",
        units="hours",
    ),    
}
search_space_names = sorted(list(set(search_space.keys())))
input_values = [
        widgets.FloatSlider(        
            description=key,
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            tooltip=val['units'],
            layout=widgets.Layout(width='400px'),
            **val,
        ) for key,val in search_space.items()
    ]
sim_settings = []
for idx in range(max_scenarios):
    subset = [widgets.FloatText(value=np.nan, readout_format='.3g', 
                                 layout=widgets.Layout(width='100px')) for _ in search_space_names]
    sim_settings.append(subset)

def generate_plots(df_dict, n_frames=100):    
    highlight = {f'{{"width": 4, "color": "{colour_order[idx]}"}}':[val] \
                 for idx, val in enumerate(df_dict.keys())}
    return plot_multitags(
        df_dict = df_dict, 
        time_column = 'Time [hours]',
        batches_to_highlight=highlight,
        settings=dict(
            nrows=5, 
            show_legend=False,
            html_image_height=1000,
            html_aspect_ratio_w_over_h=1.6,
            animate=False,           
        ),
    )



def create_user_interface(max_scenarios=5):
    """Creates the user interface"""
    
    headers = ["Input value", "Units"]
    headers.extend([f"Scenario {i+1}" for i in range(max_scenarios)])
    widths = [500, 100]
    widths.extend([200 for i in range(max_scenarios)])
    s1 = sheet(rows=len(search_space.keys()) + 1, columns=7, column_headers=headers, column_width=widths)
    column(0, input_values)
    column(1, [val['units'] for _, val in search_space.items()])
    
    for idx in range(max_scenarios):
        column(idx+2, sim_settings[idx])

    startoff = [widgets.HTML(value="<span style='float:right'><b>Colour</b></span>", placeholder='', description=''), 
                widgets.HTML(value="", placeholder='', description='')
               ]
    startoff.extend(
        [widgets.HTML(value=f"<span style='color: {colour_order[idx]}'><b>Scenario {idx+1}<b></span>") for idx in range(max_scenarios)]
    )
    row(len(search_space), startoff)
    solve_button = widgets.Button(
        description='Run simulation →'); solve_button.on_click(run_simulation); 

    box_layout = Layout(display='inline-flex', flex_flow='column', align_items='flex-start', width='100%')
    box_auto = Box(children=[s1, HBox([solve_button])],layout=box_layout)
    display(VBox([box_auto, text_out ]) );    


text_out = Output()
@text_out.capture(clear_output=True, wait=True)
def run_simulation(x_search = None):
    """ Runs the ODE integrator simulation, calculates extra columns and plots all the results."""
    
    if len(df_dict) == max_scenarios:
        print("Maximum number of scenarios reached. Stopping to avoid overwriting results.")
        return
    user_input_dict = {val.description: val.value for val in input_values}
    outputs = call_fermentation_model(settings = user_input_dict)    
    all_outputs = complete_balance_calculations(outputs, settings=user_input_dict)    
    
    df_dict[f"Scenario {len(df_dict)+1}"] = all_outputs
    vector = sim_settings[len(df_dict)-1]
    for idx, val in enumerate(user_input_dict.values()):
        vector[idx].value = val
    
    display(generate_plots(df_dict, n_frames=user_input_dict['Duration']/2))
  