import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
from PIL import Image

# Assuming images are stored in 'assets/backgrounds/' with predefined options
background_images = {
    'Background 1': 'assets/backgroundimage/09012023-DAPI.jpg',
    'Background 2': 'assets/backgroundimage/08102023-MIN_Matrix.jpg',
    # Add more backgrounds as needed
}

# Sample data for each background
scatter_data_options = {
    'Background 1': {
        'x': np.random.randn(50),
        'y': np.random.randn(50) + 2,  # Some random data
        'mode': 'markers',
        'marker': {'size': 12, 'color': 'LightSkyBlue'}
    },
    'Background 2': {
        'x': np.random.randn(50) + 1,
        'y': np.random.randn(50) + 3,  # Different random data
        'mode': 'markers',
        'marker': {'size': 12, 'color': 'Violet'}
    },
    # Add more data sets as needed
}

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='background-selector',
        options=[{'label': k, 'value': k} for k in background_images.keys()],
        value='Background 1'  # Default value
    ),
    dcc.Graph(id='scatter-plot',style={'height': '90vh', 'width': '90vw'} )
])

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('background-selector', 'value')]
)
def update_plot(selected_background):
    scatter_data = scatter_data_options[selected_background]

    # # Get the size of the selected background image
    # with Image.open(background_images[selected_background]) as img:
    #     width, height = img.size
    #     aspect_ratio = width / height
    
    # # Define the layout size based on the image size
    # # Adjusting the layout size to maintain the aspect ratio of the background
    # layout_width = 1000  # You can set this to your preferred width
    # layout_height = layout_width / aspect_ratio

    # Create scatter plot
    fig = go.Figure(data=[go.Scatter(x=scatter_data['x'], y=scatter_data['y'], mode=scatter_data['mode'], marker=scatter_data['marker'])])

    # Update figure layout to include the background image and adjust size
    fig.update_layout(
        images=[go.layout.Image(
            source=background_images[selected_background],
            xref="paper", yref="paper",
            x=0, y=1,
            sizex=1, sizey=1,
            sizing="stretch",
            opacity=0.5,
            layer="below")],
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),  # This removes the default margin to use the full area
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
