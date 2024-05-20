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
column_options = [{'label': ligand, 'value': ligand} for ligand in df['Ligand'].unique()]
row_options = [{'label': receptor, 'value': receptor} for receptor in df['Receptor'].unique()]

df_gene = pd.read_csv('./assets/data_09012023/geneexpression.csv')
geneList = [{'label': gene, 'value': gene} for gene in df_gene.columns[3:]]

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
    html.Label('Select Gene to mark:', style={'margin-right': '10px', 'align-self': 'center', 'font-size': '16px'}),
    dcc.Dropdown(
        id='Gene-dropdown',
        options=geneList,
        placeholder='Select Gene',
        style={'width': '40%', 'display': 'inline-block', 'margin-top':'10px','margin-right': '0', 'align-self': 'center'}
    ),
    html.Div([
        html.Label('Select Ligand and Receptor to plot Communicating cells:', style={'margin-right': '15px', 'align-self': 'center', 'font-size': '16px'}),
        dcc.Dropdown(
            id='Ligand-dropdown',
            options=column_options,
            placeholder='Select Ligand',
            style={'width': '35%', 'display': 'inline-block', 'margin-right': '0', 'align-self': 'center'}
        ),
        dcc.Dropdown(
            id='Receptor-dropdown',
            options=row_options,
            placeholder='Select Receptor',
            style={'width': '35%', 'display': 'inline-block', 'margin-left': '0', 'align-self': 'center'}
        )
    ],  style={'display': 'flex', 'justify-content': 'flex-start', 'align-items': 'center', 'margin-top': '20px', 'margin-bottom': '20px'}),
    html.Div(id='message-display', style={'margin-top': '20px', 'color': 'red', 'font-weight': 'bold'}),
    html.Div(id='update-info', style={'display': 'none'}),
    dcc.Graph(id='communicating-cells', style={'margin-top': '20px'}),
    html.Div(style={'bottom': '10px', 'right': '10px', 'width': '300px', 'margin-top':'10px'},
        children=[html.Img(src='./assets/images_dir/banner_footer.png', style={'width': '100%', 'height': 'auto'})]
    )
])
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('dataset-selector', 'value'),
     Input('Gene-dropdown', 'value')]
)
def update_output(selected_dataset, gene):
    if gene:
        fig = update_plot_colors(selected_dataset, gene)
        return fig
    elif selected_dataset:
        return update_plot(selected_dataset)
    return go.Figure()


@app.callback(
    [Output('communicating-cells', 'figure'),
    Output('message-display', 'children')],
    [Input('Ligand-dropdown', 'value'), 
     Input('Receptor-dropdown', 'value')]
)
def update_network(ligand, receptor):
    if not ligand or not receptor:
        return go.Figure(), "" 
    fig, msg = update_network_graph(ligand, receptor)

    return fig, msg

if __name__ == '__main__':
    app.run_server(debug=True)
