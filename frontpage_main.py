import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os
from flask import Flask
from helpers import update_plot  # Ensure this returns a proper Plotly figure

# Setup your server and app
server = Flask(__name__)
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True)

# Load data from CSV
df = pd.read_csv('./assets/data_09012023/geneexpression.csv')
column_options = [{'label': col, 'value': col} for col in df.columns[3:]]
row_options = [{'label': str(index), 'value': str(index)} for index in df.index]  # Assuming row index as identifier

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
            id='Ligent-dropdown',
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
    html.Div(id='button-container', style={'margin-top': '50px'}),
    html.Div(style={'position': 'absolute', 'bottom': '10px', 'right': '10px', 'width': '200px'},
        children=[html.Img(src='./assets/images_dir/banner_footer.png', style={'width': '100%', 'height': 'auto'})]
    )
])

@app.callback(
    [Output('scatter-plot', 'figure'),  
     Output('update-info', 'children')],
    [Input('dataset-selector', 'value')]
)
def update_figure(selected_data):
    if selected_data:
        fig = update_plot(selected_data)  # Ensure this returns a plotly figure
        return fig, "Subplots options" 
    return {}, "No data selected. Please select dataset."

# Call back for ligand and receptor

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
