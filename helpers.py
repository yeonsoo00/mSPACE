import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import os
from PIL import Image
import io
from glob import glob
import base64
import math
import networkx as nx

# Directory where datasets are stored
datasets_dir = './assets'

# Automatically list available datasets based on folder names
available_datasets = [name for name in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, name))]

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='dataset-selector',
        options=[{'label': dataset, 'value': dataset} for dataset in available_datasets],
        value=available_datasets[0] if available_datasets else None  # Default to first dataset or None
    ),
    dcc.Graph(id='scatter-plot')
])

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('dataset-selector', 'value')]
)
def update_plot(selected_dataset):
    if not selected_dataset:
        return go.Figure()
    
    # Construct paths to the dataset's CSV file and background image
    csv_file_path = os.path.join(datasets_dir, selected_dataset, 'geneexpression.csv')
    background_image_path = os.path.join(datasets_dir, selected_dataset, 'background.jpg')
    
    # Load scatter plot data from CSV
    df = pd.read_csv(csv_file_path)
    
    hovertemplate = ''
    for col in df.columns:
        hovertemplate += f'<b>{col}</b>: %{{customdata[{df.columns.get_loc(col)}]}}<br>'

    # Create scatter plot
    # when ligand and receptor are chosen, refresh the plot with links
    fig = go.Figure(data=[go.Scatter(
        x=df['cellx'], 
        y=df['celly'], 
        mode='markers',
        marker=dict(size=4,color=df.get('color', 'blue')),  
        customdata=df.values,
        hovertemplate=hovertemplate
    )])

    # Convert the background image to base64
    with Image.open(background_image_path) as img:
        with io.BytesIO() as buffer:
            img.save(buffer, format='JPEG')
            encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Update figure layout to include the background image
    fig.update_layout(
        images=[go.layout.Image(
            source=f"data:image/jpeg;base64,{encoded_image}",
            xref="x", yref="y",
            x=0, y=0,
            sizex=df['cellx'].max(),  
            sizey=df['celly'].max(),  
            sizing="stretch",  
            opacity=0.5,
            layer="below")],
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False,autorange='reversed'),
        margin=dict(l=0, r=0, t=0, b=0),  
    )

    return fig


def update_plot_colors(selected_dataset, genes):

    # CSV file and background image
    csv_file_path = os.path.join(datasets_dir, selected_dataset, 'geneexpression.csv')
    background_image_path = os.path.join(datasets_dir, selected_dataset, 'background.jpg')

    # Load scatter plot data from CSV
    df = pd.read_csv(csv_file_path)

    if genes:
        # Filter the dataframe to include only cells that express all selected genes
        mask = df[genes] > 0  # Boolean mask for cells where gene expression > 0
        all_genes_expressed = mask.all(axis=1)  # Check if all conditions across selected genes are True for each cell
        df['color'] = 'blue'  # Default color
        df.loc[all_genes_expressed, 'color'] = 'red'  # Cells expressing all genes in red
    else:
        # If no genes are selected, default all to blue
        df['color'] = 'blue'

    # Create hover template
    hovertemplate = '<br>'.join([f'{col}: %{{customdata[{i}]}}' for i, col in enumerate(df.columns)])

    # Create scatter plot
    fig = go.Figure(data=[go.Scatter(
        x=df['cellx'], 
        y=df['celly'], 
        mode='markers',
        marker=dict(size=5, color=df['color']), 
        customdata=df.values,
        hovertemplate=hovertemplate
    )])


    # Convert the background image to base64
    with Image.open(background_image_path) as img:
        with io.BytesIO() as buffer:
            img.save(buffer, format='JPEG')
            encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Update figure layout to include the background image
    fig.update_layout(
        images=[go.layout.Image(
            source=f"data:image/jpeg;base64,{encoded_image}",
            xref="x", yref="y",
            x=0, y=0,
            sizex=df['cellx'].max(),
            sizey=df['celly'].max(),
            sizing="stretch",
            opacity=0.5,
            layer="below")],
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, autorange='reversed'),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig

def update_network_graph(ligand, receptor):

    # Load datasets
    df_comm = pd.read_csv('./CLARA/Dataset/Testdata/Mapped_new/csv/Mapped_new_0.08_CCI_details.csv')  # CCI data
    df_cells = pd.read_csv('./assets/data_09012023/geneexpression.csv')  # gene expression data

    # Filter data
    filtered_comm = df_comm[(df_comm['Ligand'] == ligand) & (df_comm['Receptor'] == receptor)]

    # Build the graph
    G = nx.DiGraph()
    for _, row in df_cells.iterrows():
        G.add_node(row['cellid'], pos=(row['cellx'], row['celly']), customdata=row.values)

    for _, row in filtered_comm.iterrows():
        G.add_edge(row['Sender'], row['Receiver'])

    # Prepare to plot
    node_x = []
    node_y = []
    node_color = []
    node_customdata = []
    annotations = []

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        node_color.append('rgba(149, 165, 166, 0.5)' if node not in set(filtered_comm['Sender']) | set(filtered_comm['Receiver']) else 'rgba(52, 152, 219, 1)')
        node_customdata.append(G.nodes[node]['customdata'])

    # Define hovertemplate
    hovertemplate = '<br>'.join([f'{col}: %{{customdata[{i}]}}' for i, col in enumerate(df_cells.columns)])
    
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text',
        marker=dict(color=node_color, size=8),
        customdata=node_customdata,
        hovertemplate=hovertemplate
    )

    # Add edges as annotations
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        ax, ay, x, y = adjust_arrow_position(x0, y0, x1, y1, node_size=5)
        annotations.append(
            dict(ax=ax, ay=ay, axref='x', ayref='y', x=x, y=y, xref='x', yref='y',
                 arrowwidth=1, arrowcolor="#808080", arrowsize=1, showarrow=True, arrowhead=1)
        )

    fig = go.Figure(
        data=[node_trace],
        layout=go.Layout(
            title='Directed Graph of Communicating Cells',
            title_x=0.5,
            showlegend=False,
            hovermode='closest',
            template="simple_white",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=True, autorange='reversed'),
            annotations=annotations
        )
    )

    return fig

def adjust_arrow_position(x0, y0, x1, y1, node_size):
    # Calculate direction of the edge
    dx = x1 - x0
    dy = y1 - y0
    distance = math.hypot(dx, dy)
    if distance ==0:
        distance =0.5
    dx *= node_size / distance
    dy *= node_size / distance

    # Adjusted positions
    return x0 + dx, y0 + dy, x1 - dx, y1 - dy


if __name__ == '__main__':
    app.run_server(port = 8081, debug=True)
