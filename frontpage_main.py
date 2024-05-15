import dash
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px
import os
from dash import dcc
from dash.exceptions import PreventUpdate
from dash import dcc
from dash import no_update
import dash_core_components as dcc
from flask import Flask, redirect
from helpers import *


datasets_dir = './assets'
available_datasets = [name for name in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, name))]

server = Flask(__name__)
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True)

# Define the layout of the app with URL routing

app.layout = html.Div([
    html.H1("mSPACE (Multi-layer Spatial Articular Cartilage Explorer)", style={'font-weight': 'bold'}),
    html.Div([
        dcc.Dropdown(
            id='dataset-selector',
            options=[{'label': dataset, 'value': dataset} for dataset in available_datasets],
            placeholder='Please select existing data file',
            style={'display': 'inline-block', 'width': '80%', 'margin-top':'5px'}
        ),
    ], style={'display': 'flex', 'justify-content': 'space-between'}),
    dcc.Graph(id='scatter-plot', style={'margin-top': '20px'}),
    html.Img(src='./assets/images_dir/banner_footer.png', style={'width': '20%', 'height': 'auto', 'float': 'right', 'margin-top': '5px'})
], style={'margin': '10px'})

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('dataset-selector', 'value')
)
def update_figure(selected_data):
    # if selected_data is not None:
    return update_plot(selected_data)

if __name__ == '__main__':
    app.run_server(port=8081, debug=True)