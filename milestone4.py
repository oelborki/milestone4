from dash import dcc, html, Input, Output, State, callback
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import dash
import pandas as pd
import io
import base64

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# global variables
uploaded_file_content = None
trained_model = None
feature_order = []
categorical_columns = []

# app layout
app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload File'),
        multiple=False,
        style={
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'padding': '10px',
            'textAlign': 'center',
            'width': '50%',
            'margin': '10px auto',
        }
    ),
    html.Div(id='output-data-upload'),
    html.Div(id='dropdown-container', style={'margin': '20px auto', 'width': '50%'}),
    html.Div([
        html.Div(
            [
                html.Div(id='radio-container', style={
                    'textAlign': 'center',
                    'marginBottom': '5px'
                }),
                dcc.Graph(id='bar-chart-average')
            ],
            style={'width': '50%', 'minWidth': '300px', 'flex': '1'}
        ),

        html.Div(
            dcc.Graph(id='bar-chart-correlation'),
            style={'width': '50%', 'minWidth': '300px', 'flex': '1'}
        )
    ], style={'display': 'flex', 'flexWrap': 'wrap'}),
    html.Div([
        dcc.Checklist(
            id='feature-selection-checkbox',
            inline=True,
            labelStyle={'margin-right': '10px'}
        ),
        html.Button(
            'Train',
            id='train-button',
            n_clicks=0,
            style={
                'border': '1px solid gray',
                'padding': '5px 10px',
                'marginTop': '10px',
                'cursor': 'pointer'
            }
        ),
    ], style={'width': '50%', 'margin': '20px auto', 'textAlign': 'center'}),
    html.Div(id='r2-score-display', style={'marginTop': '20px', 'textAlign': 'center'}),
    html.Div([
        dcc.Input(id='prediction-input', type='text', placeholder="Enter input (comma-separated)", style={'marginRight': '10px'}),
        html.Button(
            'Predict',
            id='predict-button',
            n_clicks=0,
            style={
                'border': '1px solid gray',
                'padding': '5px 10px',
                'cursor': 'pointer'
            }
        ),
        html.Div(id='prediction-output', style={'marginTop': '10px'})
    ], style={'width': '50%', 'margin': '20px auto', 'textAlign': 'center'})
])

# data preprocessing
def preprocess_data(df):
    df = df.fillna(df.median(numeric_only=True))
    return df

# file upload
@callback(
    [Output('output-data-upload', 'children'),
     Output('dropdown-container', 'children'),
     Output('radio-container', 'children'),
     Output('feature-selection-checkbox', 'options')],
    Input('upload-data', 'contents')
)
def update_output(contents):
    global uploaded_file_content, categorical_columns

    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        try:
            uploaded_file_content = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            uploaded_file_content = preprocess_data(uploaded_file_content)

            numerical_cols = uploaded_file_content.select_dtypes(include=['number']).columns
            dropdown = dcc.Dropdown(
                id='target-variable-dropdown',
                options=[{'label': col, 'value': col} for col in numerical_cols],
                value=numerical_cols[0] if len(numerical_cols) > 0 else None,
                placeholder='Select a numerical target variable',
                style={'width': '100%'}
            )

            categorical_columns = uploaded_file_content.select_dtypes(include=['object', 'category']).columns
            radio_buttons = html.Div([
                                       dcc.RadioItems(
                                           id='categorical-variable-radio',
                                           options=[{'label': col, 'value': col} for col in categorical_columns],
                                           value=categorical_columns[0] if len(categorical_columns) > 0 else None,
                                           inline=True,
                                           labelStyle={'margin-right': '10px'}
                                       )]) if len(categorical_columns) > 0 else html.Div("No categorical variables available.")

            all_features = [{'label': col, 'value': col} for col in uploaded_file_content.columns]

            return html.Div(""), dropdown, radio_buttons, all_features
        except Exception as e:
            return html.Div(f"Error: {str(e)}"), None, None, []

    return html.Div(""), None, None, []

@callback(
    Output('bar-chart-average', 'figure'),
    [Input('categorical-variable-radio', 'value'),
     Input('target-variable-dropdown', 'value')] 
)
def update_bar_chart(categorical_var, target_var):
    global uploaded_file_content

    if categorical_var and target_var and uploaded_file_content is not None:
        avg_data = uploaded_file_content.groupby(categorical_var)[target_var].mean().reset_index()
        return {
            'data': [
                {
                    'x': avg_data[categorical_var],
                    'y': avg_data[target_var],
                    'type': 'bar',
                    'marker': {'color': 'indigo'}
                }
            ],
            'layout': {
                'title': f'Average {target_var} by {categorical_var}',
                'xaxis': {'title': categorical_var},
                'yaxis': {'title': f'{target_var} (average)'}
            }
        }

    return {
        'data': [],
        'layout': {
            'title': 'No data available',
            'xaxis': {'title': ''},
            'yaxis': {'title': ''}
        }
    }

@callback(
    Output('bar-chart-correlation', 'figure'),
    Input('target-variable-dropdown', 'value')
)
def update_correlation_chart(target_var):
    global uploaded_file_content

    if target_var and uploaded_file_content is not None:
        numerical_cols = uploaded_file_content.select_dtypes(include=['number']).columns
        correlations = uploaded_file_content[numerical_cols].corr()[target_var].drop(target_var).abs()
        correlation_data = correlations.sort_values(ascending=False).reset_index()
        correlation_data.columns = ['Variable', 'Correlation']

        return {
            'data': [
                {
                    'x': correlation_data['Variable'],
                    'y': correlation_data['Correlation'],
                    'type': 'bar',
                    'marker': {'color': 'mediumturquoise'}
                }
            ],
            'layout': {
                'title': f'Correlation Strength of Numerical Variables with {target_var}',
                'xaxis': {'title': 'Numerical Variables'},
                'yaxis': {'title': 'Correlation Strength (Absolute Value)'}
            }
        }

    return {}

@callback(
    Output('r2-score-display', 'children'),
    Input('train-button', 'n_clicks'),
    [State('target-variable-dropdown', 'value'),
     State('feature-selection-checkbox', 'value')]
)
def train_model(n_clicks, target_var, selected_features):
    global uploaded_file_content, trained_model, feature_order

    if n_clicks > 0 and target_var and selected_features:
        X = uploaded_file_content[selected_features]
        y = uploaded_file_content[target_var]
        feature_order = selected_features

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        numeric_features = X.select_dtypes(include=['number']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ]
        )

        param_grid = {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__max_depth': [3, 5, 10],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__subsample': [0.6, 0.8, 1.0],
        }

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', GradientBoostingRegressor(random_state=42))])

        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        trained_model = grid_search.best_estimator_

        y_pred = trained_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        return (f"R² Score: {r2:.2f}")

    return "Please select your features."

@callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [State('prediction-input', 'value'),
     State('target-variable-dropdown', 'value')]
)
def make_prediction(n_clicks, input_values, target_var):
    global trained_model, feature_order, categorical_columns

    if n_clicks > 0 and input_values and trained_model:
        
            input_values = [x.strip() for x in input_values.split(',')]
            for i, col in enumerate(feature_order):
                if col in categorical_columns:
                    input_values[i] = str(input_values[i])
                else:
                    input_values[i] = float(input_values[i])

            input_df = pd.DataFrame([input_values], columns=feature_order)
            prediction = trained_model.predict(input_df)
            return f"Predicted {target_var}: {prediction[0]:.2f}"

    return "Please enter values to make a prediction."

    
server = app.server
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
