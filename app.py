import dash
from dash import dcc,html
from dash.dependencies import Input, Output, State
import pickle
import numpy as np

# Global variables
myheading1 = 'California Housing Dataset'
tabtitle = 'Cali Housing'
sourceurl = 'https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb'
githublink = 'https://github.com/cahn1/502-california-housing-regression/tree/update1'


# Load pkl files
with open('analysis/model_components/coefs_fig.pkl', 'rb') as f:
    coefs = pickle.load(f)
with open('analysis/model_components/r2_fig.pkl', 'rb') as f:
    r2_fig = pickle.load(f)
with open('analysis/model_components/rmse_fig.pkl', 'rb') as f:
    rmse_fig = pickle.load(f)
with open('analysis/model_components/std_scaler.pkl', 'rb') as f:
    std_scaler = pickle.load(f)
with open('analysis/model_components/lin_reg.pkl', 'rb') as f:
    lin_reg = pickle.load(f)
with open('analysis/model_components/knn_model.pkl', 'rb') as f:
    knn = pickle.load(f)
with open('analysis/model_components/random_forest.pkl', 'rb') as f:
    random_forest = pickle.load(f)

# app server config
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = tabtitle

# Setup a layout
app.layout = html.Div([
    html.H1('Regression Analysis', style={
        'textAlign': 'center', 'margin': '48px 0', 'fontFamily': 'system-ui'}),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='California Neighborhoods: Linear Regression', children=[
            # Prediction Block
            html.Div(children=[
                html.Div([
                    html.Div('Longitude:'),
                    dcc.Input(id='longitude', value=-119.5, type='number', min=-124.3, max=-114.3, step=.1),

                    html.Div('Latitude:'),
                    dcc.Input(id='latitude', value=35.6, type='number', min=32.5, max=41.95, step=.1),

                    html.Div('Housing Median Age:'),
                    dcc.Input(id='housing_median_age', value=28, type='number', min=1, max=52, step=1),

                    html.Div('Total Rooms:'),
                    dcc.Input(id='total_rooms', value=2000, type='number', min=1000, max=3000, step=100),

                    html.Div('Population:'),
                    dcc.Input(id='population', value=1500, type='number', min=1000, max=35000, step=500),
                ], className='four columns'),
                html.Div([
                    html.Div('Households:'),
                    dcc.Input(id='households', value=1000, type='number', min=500, max=6000, step=500),

                    html.Div('Median Home Income:'),
                    dcc.Input(id='median_income', value=2, type='number', min=1, max=15, step=1),

                    html.Div('Income Category:'),
                    dcc.Input(id='income_cat', value=3, type='number', min=1, max=5, step=1),

                    html.Div('Rooms per Household:'),
                    dcc.Input(id='rooms_per_hhold', value=5, type='number', min=1, max=7, step=1),

                    html.Div('Population per Household:'),
                    dcc.Input(id='pop_per_household', value=3, type='number', min=1, max=10, step=1),
                ], className='four columns'),
                html.Div([
                    html.H6('Median Home Value (Predicted):'),
                    # html.Button(children='Submit', id='submit-val', n_clicks=0,
                    #             style={'background-color': 'red',
                    #                    'color': 'white',
                    #                    'margin-left': '5px',
                    #                    'verticalAlign': 'center',
                    #                    'horizontalAlign': 'center'}
                    #             ),
                    # html.Div(id='Results')
                    # dcc.Dropdown(['Linear Regression', 'KNN', 'Random Forest'],
                    #              'Linear Regression', id='regression_type'),
                    dcc.Dropdown(
                        options=[
                            {'label': 'Linear Regression', 'value': 'Linear Regression'},
                            {'label': 'KNN', 'value': 'KNN'},
                            {'label': 'Random Forest', 'value': 'Random Forest'},
                        ],
                        value='Linear Regression',
                        id='regression_type',
                        style={
                            'background-color': '#f7dc6f',
                        }
                    ),
                    html.Div(id='output-container1')
                ], className='four columns')
            ], className='twelve columns'),

            # Evaluation Block
            html.Div(children=[
                html.Div([
                    dcc.Graph(figure=r2_fig, id='r2_fig')], className='six columns'),
                html.Div([
                    dcc.Graph(figure=rmse_fig, id='rmse_fig')], className='six columns'),
            ], className='twelve columns'),

            html.Div(children=[
                html.H3('Linear Regression Coefficients (standardized features)'),
                dcc.Graph(figure=coefs, id='coefs_fig')
            ], className='twelve columns'),
            html.A('Code on Github', href=githublink),
            html.Br(),
            html.A("Data Source", href=sourceurl),
        ]),
        # dcc.Tab(label='Random Forest', children=[
        #     # Prediction Block
        #     html.Div(children=[
        #         html.Div([
        #             html.Div('Longitude:'),
        #             dcc.Input(id='longitude1', value=-119.5, type='number',
        #                       min=-124.3, max=-114.3, step=.1),
        #
        #             html.Div('Latitude:'),
        #             dcc.Input(id='latitude1', value=35.6, type='number',
        #                       min=32.5, max=41.95, step=.1),
        #
        #             html.Div('Housing Median Age:'),
        #             dcc.Input(id='housing_median_age1', value=28,
        #                       type='number', min=1, max=52, step=1),
        #
        #             html.Div('Total Rooms:'),
        #             dcc.Input(id='total_rooms1', value=2000, type='number',
        #                       min=1000, max=3000, step=100),
        #
        #             html.Div('Population:'),
        #             dcc.Input(id='population1', value=1500, type='number',
        #                       min=1000, max=35000, step=500),
        #         ], className='four columns'),
        #         html.Div([
        #             html.Div('Households:'),
        #             dcc.Input(id='households1', value=1000, type='number',
        #                       min=500, max=6000, step=500),
        #
        #             html.Div('Median Home Income:'),
        #             dcc.Input(id='median_income1', value=2, type='number',
        #                       min=1, max=15, step=1),
        #
        #             html.Div('Income Category:'),
        #             dcc.Input(id='income_cat1', value=3, type='number',
        #                       min=1, max=5, step=1),
        #
        #             html.Div('Rooms per Household:'),
        #             dcc.Input(id='rooms_per_hhold1', value=5, type='number',
        #                       min=1, max=7, step=1),
        #
        #             html.Div('Population per Household:'),
        #             dcc.Input(id='pop_per_household1', value=3,
        #                       type='number', min=1, max=10, step=1),
        #         ], className='four columns'),
        #         html.Div([
        #             html.H6('Median Home Value (Predicted):'),
        #             html.Button(children='Submit', id='submit-val1', n_clicks=0,
        #                         style={'background-color': 'red',
        #                                'color': 'white',
        #                                'margin-left': '5px',
        #                                'verticalAlign': 'center',
        #                                'horizontalAlign': 'center'}
        #                         ),
        #             html.Div(id='Results1')
        #         ], className='four columns')
        #     ], className='twelve columns'),
        # ]),
        # dcc.Tab(label='Tab three', children=[
        #     html.Div([
        #         html.H1("This is the content in tab 3"),
        #         dcc.Graph(
        #             id='example-graph2',
        #             figure={
        #                 'data': [
        #                     {'x': [1, 2, 3], 'y': [5, 2, 3],
        #                      'type': 'scatter', 'name': 'SF'},
        #                     {'x': [1, 2, 3], 'y': [6, 8, 1],
        #                      'type': 'scatter', 'name': u'Montréal'},
        #                 ],
        #                 'layout': {
        #                     'title': '2 Dash Data Visualization'
        #                 }
        #             }
        #         )
        #     ])
        # ]),
    ],
     style={
         'fontFamily': 'system-ui'
     },
     content_style={
         'borderLeft': '1px solid #d6d6d6',
         'borderRight': '1px solid #d6d6d6',
         'borderBottom': '1px solid #d6d6d6',
         'padding': '44px'
     },
     parent_style={
         'maxWidth': '1400px',
         'margin': '0 auto'
     })
])


# callback
@app.callback(
    Output(component_id='output-container1', component_property='children'),
    Input('regression_type', 'value'),
    # Regression inputs:
    State(component_id='longitude', component_property='value'),
    State(component_id='latitude', component_property='value'),
    State(component_id='housing_median_age', component_property='value'),
    State(component_id='total_rooms', component_property='value'),
    State(component_id='population', component_property='value'),
    State(component_id='households', component_property='value'),
    State(component_id='median_income', component_property='value'),
    State(component_id='income_cat', component_property='value'),
    State(component_id='rooms_per_hhold', component_property='value'),
    State(component_id='pop_per_household', component_property='value'),
)
def make_prediction(
        value,
        longitude,
        latitude,
        housing_median_age,
        total_rooms,
        population,
        households,
        median_income,
        income_cat,
        rooms_per_hhold,
        pop_per_household):
    inputs = np.array([
        longitude,
        latitude,
        housing_median_age,
        total_rooms,
        population,
        households,
        median_income,
        income_cat,
        rooms_per_hhold,
        pop_per_household, 0, 0, 0, 0]).reshape(1, -1)
    # Standardization
    std_inputs = std_scaler.transform(inputs)

    y = lin_reg.predict(std_inputs) if value == 'Linear Regression' else \
        random_forest.predict(std_inputs) if value == 'Random Forest' else \
            knn.predict(std_inputs)
    return f'${y[0]:,.2f}'


# # callback
# @app.callback(
#     Output(component_id='Results1', component_property='children'),
#     Input(component_id='submit-val1', component_property='n_clicks'),
#     # Regression inputs:
#     State(component_id='longitude1', component_property='value'),
#     State(component_id='latitude1', component_property='value'),
#     State(component_id='housing_median_age1', component_property='value'),
#     State(component_id='total_rooms1', component_property='value'),
#     State(component_id='population1', component_property='value'),
#     State(component_id='households1', component_property='value'),
#     State(component_id='median_income1', component_property='value'),
#     State(component_id='income_cat1', component_property='value'),
#     State(component_id='rooms_per_hhold1', component_property='value'),
#     State(component_id='pop_per_household1', component_property='value'),
# )
# def make_prediction(
#         clicks,
#         longitude1,
#         latitude1,
#         housing_median_age1,
#         total_rooms1,
#         population1,
#         households1,
#         median_income1,
#         income_cat1,
#         rooms_per_hhold1,
#         pop_per_household1):
#     if clicks == 0:
#         return "waiting for inputs"
#     else:
#         inputs = np.array([
#             longitude1,
#             latitude1,
#             housing_median_age1,
#             total_rooms1,
#             population1,
#             households1,
#             median_income1,
#             income_cat1,
#             rooms_per_hhold1,
#             pop_per_household1, 0, 0, 0, 0]).reshape(1, -1)
#         # note: the 4 zeroes are for missing categories=['INLAND', 'ISLAND',
#         # 'NEAR BAY','NEAR OCEAN']
#         # test with fake inputs
#         # fake = np.array([-122, 37, 40, 2000, 3000, 500, 3, 3, 6, 4, 0, 0,
#         # 1, 0]).reshape(1, -1)
#         # std_fake = std_scaler.transform(fake)
#         # Standardization
#         std_inputs = std_scaler.transform(inputs)
#
#         y = random_forest.predict(std_inputs)
#         return f'${y[0]:,.2f}'


if __name__ == '__main__':
    app.run_server(debug=True)