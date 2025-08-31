# Import necessary libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import joblib
import numpy as np

# Load the dataset
file_path = 'BI - California Housing Dataset.csv'
df = pd.read_csv(file_path, delimiter=';')

# Drop rows with missing target values (median_house_value)
df.dropna(subset=['median_house_value'], inplace=True)

# Preprocessing for the model
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)
df['rooms_per_person'] = df['total_rooms'] / df['population']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']

# Load the pre-trained model
model = joblib.load('california_house_price_model.joblib')

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Define the layout of the app
app.layout = html.Div(style={'backgroundColor': '#1E1E1E', 'color': 'white', 'font-family': 'Arial, sans-serif'}, children=[
    html.Div([
        html.H1("California Property Insights Dashboard", style={'textAlign': 'center', 'margin-bottom': '10px'}),
        html.P("Analyze and predict California housing data with interactive visualizations and a machine learning model.", style={'textAlign': 'center', 'margin-bottom': '40px'})
    ]),
    
    # Main content area with two columns
    html.Div(style={'display': 'flex', 'flex-direction': 'row', 'gap': '20px', 'padding': '20px'}, children=[
        
        # Left column for filters and prediction
        html.Div(style={'width': '25%', 'padding': '20px', 'backgroundColor': '#282828', 'border-radius': '10px'}, children=[
            html.H3("Filters and Prediction", style={'border-bottom': '2px solid white', 'padding-bottom': '10px'}),
            
            # Filters section
            html.Div([
                html.Label("Population"),
                dcc.RangeSlider(
                    id='population-slider',
                    min=df['population'].min(),
                    max=df['population'].max(),
                    value=[df['population'].min(), df['population'].max()],
                    marks={i: str(i) for i in range(int(df['population'].min()), int(df['population'].max()) + 1, 5000)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Label("Median Income"),
                dcc.RangeSlider(
                    id='median-income-slider',
                    min=df['median_income'].min(),
                    max=df['median_income'].max(),
                    value=[df['median_income'].min(), df['median_income'].max()],
                    marks={i: f'${i/1000:.0f}k' for i in range(int(df['median_income'].min()), int(df['median_income'].max()) + 1, 20000)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Label("Median House Value"),
                dcc.RangeSlider(
                    id='median-house-value-slider',
                    min=df['median_house_value'].min(),
                    max=df['median_house_value'].max(),
                    value=[df['median_house_value'].min(), df['median_house_value'].max()],
                    marks={i: f'${i/1000:.0f}k' for i in range(int(df['median_house_value'].min()), int(df['median_house_value'].max()) + 1, 100000)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Label("Ocean Proximity"),
                dcc.Dropdown(
                    id='ocean-proximity-dropdown',
                    options=[{'label': i, 'value': i} for i in df['ocean_proximity'].unique()],
                    value=list(df['ocean_proximity'].unique()),
                    multi=True,
                    style={'color': 'black'}
                ),
            ], style={'margin-bottom': '40px'}),
            
            # Prediction section
            html.H4("Predict House Value", style={'border-bottom': '1px solid white', 'padding-bottom': '5px', 'margin-top': '20px'}),
            html.Div(id='prediction-inputs', children=[
                dcc.Input(id='longitude', type='number', placeholder='Longitude', style={'margin': '5px', 'width': 'calc(50% - 10px)'}),
                dcc.Input(id='latitude', type='number', placeholder='Latitude', style={'margin': '5px', 'width': 'calc(50% - 10px)'}),
                dcc.Input(id='housing_median_age', type='number', placeholder='Housing Median Age', style={'margin': '5px', 'width': 'calc(50% - 10px)'}),
                dcc.Input(id='total_rooms', type='number', placeholder='Total Rooms', style={'margin': '5px', 'width': 'calc(50% - 10px)'}),
                dcc.Input(id='total_bedrooms', type='number', placeholder='Total Bedrooms', style={'margin': '5px', 'width': 'calc(50% - 10px)'}),
                dcc.Input(id='population', type='number', placeholder='Population', style={'margin': '5px', 'width': 'calc(50% - 10px)'}),
                dcc.Input(id='households', type='number', placeholder='Households', style={'margin': '5px', 'width': 'calc(50% - 10px)'}),
                dcc.Input(id='median_income_pred', type='number', placeholder='Median Income', style={'margin': '5px', 'width': 'calc(50% - 10px)'}),
                dcc.Dropdown(
                    id='ocean_proximity_pred',
                    options=[{'label': i, 'value': i} for i in df['ocean_proximity'].unique()],
                    placeholder='Ocean Proximity',
                    style={'margin': '5px', 'width': 'calc(100% - 10px)', 'color': 'black'}
                ),
            ]),
            html.Button('Predict', id='predict-button', n_clicks=0, style={'margin-top': '20px', 'width': '100%'}),
            html.Div(id='prediction-output', style={'margin-top': '20px', 'font-size': '20px', 'text-align': 'center'})
        ]),
        
        # Right column for visualizations
        html.Div(style={'width': '75%'}, children=[
            dcc.Tabs(id="tabs", value='tab-1', children=[
                dcc.Tab(label='Market Value Map', value='tab-1', style={'backgroundColor': '#282828', 'color': 'white'}, selected_style={'backgroundColor': '#1E1E1E', 'color': 'white', 'borderTop': '2px solid #1E90FF'}),
                dcc.Tab(label='Distribution Insights', value='tab-2', style={'backgroundColor': '#282828', 'color': 'white'}, selected_style={'backgroundColor': '#1E1E1E', 'color': 'white', 'borderTop': '2px solid #1E90FF'}),
                dcc.Tab(label='Correlation Analysis', value='tab-3', style={'backgroundColor': '#282828', 'color': 'white'}, selected_style={'backgroundColor': '#1E1E1E', 'color': 'white', 'borderTop': '2px solid #1E90FF'})
            ], style={'height': '48px'}),
            html.Div(id='tabs-content', style={'padding': '20px', 'backgroundColor': '#282828', 'border-radius': '10px', 'margin-top': '10px'})
        ])
    ])
])

# Callback to render tab content
@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            dcc.Graph(id='scatter-geo', style={'height': '80vh'})
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.Label("Select feature for distribution plot:"),
            dcc.Dropdown(
                id='dist-feature-dropdown',
                options=[{'label': col, 'value': col} for col in df.columns if df[col].dtype in ['int64', 'float64']],
                value='median_house_value',
                style={'color': 'black'}
            ),
            dcc.Graph(id='distribution-plot')
        ])
    elif tab == 'tab-3':
        return html.Div([
            dcc.Graph(id='correlation-heatmap', style={'height': '80vh'})
        ])

# Update graph callback
@app.callback(
    Output('scatter-geo', 'figure'),
    [Input('population-slider', 'value'),
     Input('median-income-slider', 'value'),
     Input('median-house-value-slider', 'value'),
     Input('ocean-proximity-dropdown', 'value')]
)
def update_graph(population_range, median_income_range, median_house_value_range, ocean_proximity):
    # Filter the data based on the slider ranges and dropdown selection
    filtered_df = df[
        (df['population'] >= population_range[0]) & (df['population'] <= population_range[1]) &
        (df['median_income'] >= median_income_range[0]) & (df['median_income'] <= median_income_range[1]) &
        (df['median_house_value'] >= median_house_value_range[0]) & (df['median_house_value'] <= median_house_value_range[1])
    ]
    if ocean_proximity:
        filtered_df = filtered_df[filtered_df['ocean_proximity'].isin(ocean_proximity)]

    # Create the scatter geo plot
    fig = px.scatter_geo(
        filtered_df,
        lat='latitude',
        lon='longitude',
        color='median_house_value',
        color_continuous_scale='Viridis',
        scope='usa',
        hover_name='ocean_proximity',
        hover_data=['median_house_value', 'median_income'],
        labels={'median_house_value': 'Median House Value', 'median_income': 'Median Income'},
        template='plotly_dark'
    )

    # Update the layout of the map
    fig.update_layout(
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            showland=True,
            landcolor='rgb(217, 217, 217)',
            subunitcolor='rgb(255, 255, 255)',
            countrycolor='rgb(255, 255, 255)',
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend_title_text='Median House Value',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.1,
            xanchor='center',
            x=0.5
        )
    )

    return fig

# Callback for distribution plot
@app.callback(
    Output('distribution-plot', 'figure'),
    Input('dist-feature-dropdown', 'value')
)
def update_distribution_plot(selected_feature):
    fig = px.histogram(df, x=selected_feature, title=f'Distribution of {selected_feature}', template='plotly_dark')
    fig.update_layout(bargap=0.2, title_x=0.5)
    return fig

# Callback for correlation heatmap
@app.callback(
    Output('correlation-heatmap', 'figure'),
    Input('tabs', 'value') # Triggered when the tab is selected
)
def update_heatmap(tab_value):
    if tab_value == 'tab-3':
        numeric_cols = df.select_dtypes(include=np.number)
        corr = numeric_cols.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", template='plotly_dark', title='Feature Correlation Matrix')
        fig.update_layout(title_x=0.5)
        return fig
    return dash.no_update

# Prediction callback
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('longitude', 'value'),
     State('latitude', 'value'),
     State('housing_median_age', 'value'),
     State('total_rooms', 'value'),
     State('total_bedrooms', 'value'),
     State('population', 'value'),
     State('households', 'value'),
     State('median_income_pred', 'value'),
     State('ocean_proximity_pred', 'value')]
)
def update_prediction(n_clicks, longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity):
    if n_clicks > 0:
        try:
            # Create a dataframe for the input
            input_data = pd.DataFrame({
                'longitude': [longitude],
                'latitude': [latitude],
                'housing_median_age': [housing_median_age],
                'total_rooms': [total_rooms],
                'total_bedrooms': [total_bedrooms],
                'population': [population],
                'households': [households],
                'median_income': [median_income],
                'ocean_proximity': [ocean_proximity]
            })

            # Preprocessing for the input data
            input_data['rooms_per_person'] = input_data['total_rooms'] / input_data['population']
            input_data['bedrooms_per_room'] = input_data['total_bedrooms'] / input_data['total_rooms']
            input_data['population_per_household'] = input_data['population'] / input_data['households']
            
            # One-hot encode ocean_proximity
            input_data = pd.get_dummies(input_data, columns=['ocean_proximity'], drop_first=True)
            
            # Align columns with the model's training columns
            model_cols = model.feature_names_in_
            input_data = input_data.reindex(columns=model_cols, fill_value=0)

            # Predict
            prediction = model.predict(input_data)
            
            return f'Predicted House Value: ${prediction[0]:,.2f}'
        except Exception as e:
            return f'Error: {e}. Please check your inputs.'
    return ''

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
