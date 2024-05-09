import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import os
from helpers import *

datasets_dir = '/home/yec23006/projects/research/app/mSpace/assets' #'./assets'
available_datasets = [name for name in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, name))]

app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("mSPACE (Multi-layer Spatial Articular Cartilage Explorer)", style={'font-weight': 'bold'}),
    dcc.Dropdown(
        id='dataset-selector',
        multi=False,
        placeholder='Please select existing data file',
        options=[{'label': dataset, 'value': dataset} for dataset in available_datasets]
    ),
    dcc.Graph(id='plot')
])

# Define the callback to update the graph
@app.callback(
    Output('plot', 'figure'),
    Input('dataset-selector', 'value')
)
def update_figure(selected_data):
    # if selected_data is not None:
    return update_plot(selected_data)
# Run the app
if __name__ == '__main__':
    app.run_server(port=8081, debug=True)
