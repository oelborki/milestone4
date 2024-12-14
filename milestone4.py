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
import os

app = dash.Dash(__name__, suppress_callback_exceptions=True)
# global variables
uploaded_file_content = None
trained_model = None
feature_order = []
categorical_columns = []

# app layout
app.layout = html.Div([
    dcc.Store(id='uploaded-data-store'),  # Store for uploaded data
    dcc.Store(id='trained-model-store'),  # Store for trained model and metadata
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
     Output('feature-selection-checkbox', 'options'),
     Output('uploaded-data-store', 'data')],
    Input('upload-data', 'contents')
)
def update_output(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            df = preprocess_data(df)

            numerical_cols = df.select_dtypes(include=['number']).columns
            dropdown = dcc.Dropdown(
                id='target-variable-dropdown',
                options=[{'label': col, 'value': col} for col in numerical_cols],
                value=numerical_cols[0] if len(numerical_cols) > 0 else None,
                placeholder='Select a numerical target variable',
                style={'width': '100%'}
            )

            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            radio_buttons = html.Div([
                dcc.RadioItems(
                    id='categorical-variable-radio',
                    options=[{'label': col, 'value': col} for col in categorical_cols],
                    value=categorical_cols[0] if len(categorical_cols) > 0 else None,
                    inline=True,
                    labelStyle={'margin-right': '10px'}
                )
            ]) if len(categorical_cols) > 0 else html.Div("No categorical variables available.")

            all_features = [{'label': col, 'value': col} for col in df.columns]

            return html.Div("File uploaded successfully!"), dropdown, radio_buttons, all_features, df.to_dict('records')

        except Exception as e:
            return html.Div(f"Error: {str(e)}"), None, None, [], None

    return html.Div("No file uploaded yet."), None, None, [], None


@callback(
    Output('bar-chart-average', 'figure'),
    [Input('categorical-variable-radio', 'value'),
     Input('target-variable-dropdown', 'value'),
     State('uploaded-data-store', 'data')]
)
def update_bar_chart(categorical_var, target_var, uploaded_data):
    if uploaded_data is None:
        return {'data': [], 'layout': {'title': 'No data available'}}

    df = pd.DataFrame(uploaded_data)

    if categorical_var not in df.columns or target_var not in df.columns:
        return {'data': [], 'layout': {'title': 'Invalid column selection'}}

    avg_data = df.groupby(categorical_var)[target_var].mean().reset_index()

    return {
        'data': [{'x': avg_data[categorical_var], 'y': avg_data[target_var], 'type': 'bar'}],
        'layout': {'title': f'Average {target_var} by {categorical_var}'}
    }


@callback(
    Output('bar-chart-correlation', 'figure'),
    [Input('target-variable-dropdown', 'value'),
     State('uploaded-data-store', 'data')]
)
def update_correlation_chart(target_var, uploaded_data):
    if uploaded_data is None:
        return {'data': [], 'layout': {'title': 'No data available'}}

    df = pd.DataFrame(uploaded_data)

    if target_var not in df.columns:
        return {'data': [], 'layout': {'title': 'Invalid target variable'}}

    numerical_cols = df.select_dtypes(include=['number']).columns
    correlations = df[numerical_cols].corr()[target_var].drop(target_var).abs()
    correlation_data = correlations.sort_values(ascending=False).reset_index()
    correlation_data.columns = ['Variable', 'Correlation']

    return {
        'data': [{'x': correlation_data['Variable'], 'y': correlation_data['Correlation'], 'type': 'bar'}],
        'layout': {'title': f'Correlation with {target_var}'}
    }


@callback(
    Output('r2-score-display', 'children'),
    Output('trained-model-store', 'data'),
    Input('train-button', 'n_clicks'),
    [State('target-variable-dropdown', 'value'),
     State('feature-selection-checkbox', 'value'),
     State('uploaded-data-store', 'data')]
)
def train_model(n_clicks, target_var, selected_features, uploaded_data):
    if n_clicks > 0 and target_var and selected_features and uploaded_data:
        df = pd.DataFrame(uploaded_data)
        X = df[selected_features]
        y = df[target_var]

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

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', GradientBoostingRegressor(random_state=42))])

        param_grid = {
            'regressor__n_estimators': [50, 100],
            'regressor__max_depth': [3, 5],
            'regressor__learning_rate': [0.01, 0.05, 0.1]
        }

        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        trained_model = grid_search.best_estimator_
        y_pred = trained_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        model_data = {
            'trained_model': trained_model,
            'feature_order': selected_features,
            'categorical_columns': list(categorical_features)
        }

        return f"R² Score: {r2:.2f}", model_data

    return "Please select your features.", None


@callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [State('prediction-input', 'value'),
     State('target-variable-dropdown', 'value'),
     State('trained-model-store', 'data')]
)
def make_prediction(n_clicks, input_values, target_var, model_data):
    if n_clicks > 0 and input_values and model_data:
        trained_model = model_data['trained_model']
        feature_order = model_data['feature_order']
        categorical_columns = model_data['categorical_columns']

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
port = int(os.environ.get("PORT", 8050))  # Use PORT if available, default to 8050
app.run_server(debug=False, port=port, host="0.0.0.0")
