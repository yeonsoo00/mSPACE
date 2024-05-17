import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os
from flask import Flask
from helpers import *

# Setup your server and app
server = Flask(__name__)
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True)

# Load data from CSV
df = pd.read_csv('./CLARA/Dataset/Testdata/Mapped_new/csv/Mapped_new_0.08_CommunicationWithAxis_Y_details.csv')
column_options = [{'label': ligand, 'value': ligand} for ligand in df['Ligand']]
row_options = [{'label': receptor, 'value': receptor} for receptor in df['Receptor']]

# Define your layout
app.layout = html.Div([
    html.H1("mSPACE (Multi-layer Spatial Articular Cartilage Explorer)", style={'font-weight': 'bold'}),
    dcc.Dropdown(
        id='dataset-selector',
        options=[{'label': dataset, 'value': dataset} for dataset in os.listdir('./assets') if os.path.isdir(os.path.join('./assets', dataset))],
        placeholder='Please select existing data file',
        style={'display': 'inline-block', 'width': '80%', 'margin-top': '5px'}
    ),
    dcc.Graph(id='scatter-plot', style={'margin-top': '20px'}),
    html.Div([
        html.Label(' Select Ligand and Receptor:', style={'margin-right': '10px', 'align-self': 'center', 'font-size': '16px'}),
        dcc.Dropdown(
            id='Ligand-dropdown',
            options=column_options,
            placeholder='Select Column',
            style={'width': '40%', 'display': 'inline-block', 'margin-right': '0', 'align-self': 'center'}
        ),
        dcc.Dropdown(
            id='Receptor-dropdown',
            options=row_options,
            placeholder='Select Row',
            style={'width': '40%', 'display': 'inline-block', 'margin-left': '0', 'align-self': 'center'}
        )
    ],  style={'display': 'flex', 'justify-content': 'flex-start', 'align-items': 'center', 'margin-top': '20px', 'margin-bottom': '20px'}),
    html.Div(id='update-info', style={'display': 'none'}),
    html.Div(id='message-display', style={'margin-top': '20px', 'color': 'red', 'font-weight': 'bold'}),
    html.Div(id='button-container', style={'margin-top': '50px'}),
    html.Div(style={'position': 'absolute', 'bottom': '10px', 'right': '10px', 'width': '200px'},
        children=[html.Img(src='./assets/images_dir/banner_footer.png', style={'width': '100%', 'height': 'auto'})]
    )
])
@app.callback(
    [Output('scatter-plot', 'figure'),
    Output('message-display', 'children')],
    [Input('dataset-selector', 'value'),
     Input('Ligand-dropdown', 'value'),
     Input('Receptor-dropdown', 'value')]
)
def update_output(selected_dataset, ligand, receptor):
    # Check if Ligand and Receptor are selected to refine the plot
    if ligand and receptor:
        fig, message = update_plot_colors(selected_dataset, ligand, receptor)
        if message == "No pair matches":
            return message
        else:
            return fig, message
    elif selected_dataset:
        # Update with initial dataset selection
        return update_plot(selected_dataset)
    return go.Figure()


@app.callback(
    Output('button-container', 'children'),
    [Input('update-info', 'children')]
)
def display_buttons(info):
    if info == "Subplots options":
        return html.Div([
            html.Button("Communicated cells", id='button1', n_clicks=0, style={'margin-right': '10px'}),
            html.Button("Button 2", id='button2', n_clicks=0)
        ], style={'display': 'flex', 'justify-content': 'center', 'margin-top': '10px'})
    return []

if __name__ == '__main__':
    app.run_server(debug=True)
