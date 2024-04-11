import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import os
from PIL import Image
import io
import base64

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
    fig = go.Figure(data=[go.Scatter(
        x=df['cellx'], 
        y=df['celly'], 
        mode='markers',
        marker=dict(size=5,color=df.get('color', 'blue')),  # Adjust the size to be smaller
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
            sizex=df['cellx'].max(),  # Set the image size in x-coordinate to span the x data range
            sizey=df['celly'].max(),  # Set the image size in y-coordinate to span the y data range
            sizing="stretch",  # Maintain the original size and aspect ratio
            opacity=0.5,
            layer="below")],
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False,autorange='reversed'),
        margin=dict(l=0, r=0, t=0, b=0),  # Remove margins to fill the plot area
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
