'''
@author: Osmar Jrz

This app explore data from a file and export the results into a dashboard
to understand the key results from the End to End Data Mining process.

'''
# for manage data 
import pandas as pd 
import numpy as np
#for web dashboard 
import dash 
import dash_core_components as dcc
import dash_html_components as html
import dash_table 
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
#manage os system, paths, dates and streamed data
import os
import pathlib
import base64
import io
import datetime
#to use algorithm K Means with elbow method 
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
#to use and generate a Logistic Regression Model 
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#global variables
df = pd.DataFrame() #empty dataframe with the original data loaded
df_n = pd.DataFrame() #empty dataframe only with the numeric columns
df_kmeans = pd.DataFrame() #empty dataframe for kmeans algorithm 
df_cluster = pd.DataFrame() #empty dataframe for kmeans output - clustering out
model_vars = pd.DataFrame() #empty dataframe to store the model predict variables 
predict_vars = pd.DataFrame() #empty dataframe to store dependent variables 
final_model = None #global variable to store the clasification model obtained

app  = dash.Dash(eager_loading=True)
app.title = "Data Mining APP"


def read_file(file):
    '''
    Funtion to read some file using pandas library.

    Parameters
    ----------
    file : Str
        Path and file to read and load to the APP

    Returns
    -------
    df : Pandas Dataframe 
        Dataset loaded as dataframe from the file readed. 

    '''
    if '.csv' in str(file):
        df = pd.read_csv(file)
    elif '.xlsx' in str(file):
        df = pd.read_excel(file)
    elif '.txt' in str(file):
        df = pd.read_table(file)
    
    return df

def dd_select_dataset():
    '''
    Funtion to display the data avaible for the project loaded in the data 
    directory.

    Returns
    -------
    file_list : Lits
        List of the files loaded by default for this project.

    '''
    app_path = str(pathlib.Path(__file__).parent.resolve())
    data_path = app_path + '\\data'
    
    file_list = list()
    for file in os.listdir(data_path):
        file_list.append({'label':file, 'value':data_path + '\\' + file})
                 
    return file_list


def parse_contents(contents, filename, date):
    global df
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_header={
                        'backgroundColor': 'black',
                        'textAlign': 'center'
                    },
            style_cell={
                    'backgroundColor':'#1e2130',
                    'color': 'white',
                    'padding': '10px',
                    'textAlign': 'center'
                },
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


def missing_null_val():
    '''
    Function to build a dataframe with hey general info applying a quick 
    exploration data 

    Returns
    -------
    df_summary : Pandas Dtaframe 
        Dataframe with the generla info to display as table in EDA module.

    '''
    df_summary = pd.DataFrame(df.isna().sum(), columns = ['Count Null Values'])
    df_summary = pd.DataFrame(df.isna().sum(), columns = ['Count Null Values'])
    df_summary['Nunique values'] = df.nunique().to_list()
    df_summary['Count Records'] = df.count().to_list()
    df_summary['Columns'] = df_summary.index.to_list()
    df_summary['Data Type'] = [str(df.dtypes[col]) for col in df.columns.to_list()]
    
    df_summary = df_summary[['Columns','Count Records','Count Null Values',
                             'Nunique values','Data Type']]

    return df_summary

def desc_data(type_desc):
    '''
    Funtion to return the output dataframe by the describe function.
    
    Parameters
    ----------
    type_desc : Str
        Flag to build a describe numeric function or describe object function


    Returns
    -------
    df_desc : Pandas Dataframe
        Pandas Dtaframe table with the general info for describe method applied.

    '''
    df_desc = df.describe(include=type_desc)
    df_desc.reset_index(inplace=True)
    return df_desc


def multiple_scatter_graphs(xaxis_name):
    '''
    Funtion to display scatter graphs depending on the column selected.
    Column selected VS rest of the columns

    Parameters
    ----------
    xaxis_name : Str
        Column selected by the user in the dropdown selection component.

    Returns
    -------
    list_charts : List
        List of figures created.

    '''
    #df_n = df[df.describe().columns.to_list()]
    list_charts = []
    for col in df_n.columns:
        if col != xaxis_name:
            list_charts.append(dcc.Graph(id='fig-scatter-{}'.format(col),
                                         figure={'data': [go.Scatter(
                                                    x=df_n[xaxis_name],
                                                    y=df_n[col],
                                                    mode='markers',
                                                    marker={
                                                        'size': 15,
                                                        'opacity': 0.6,
                                                        'line': {'width':0.5, 
                                                                 'color':'white'}
                                                        }
                                                    )],
                                                'layout': go.Layout(
                                                    xaxis={'title': xaxis_name},
                                                    yaxis={'title': col},
                                                    hovermode='closest',
                                                    height=700,
                                                    width=1000
                                                )
                                            }
                                         )
                               )
            list_charts.append(html.Br())
            
    return list_charts


def kmeans_algorithm():
    '''
    This function performs all the steps in order to apply K Means Algorithm.
    The outputs generated by this function will be displaying in the Clustering
    dashboard.

    Returns
    -------
    fig_sse : Plotly Express Figure
        Line figure to show the SSE Elbow Method.
        
    fig_kl : Plotly Expresss Figure
        Line figure with the Knee point.
        
    df_cluster_desc : Pandas Dataframe
        Dataframe with random data for each clauster selected.
        
    figure_3d : Plotly Figure
        3D figure with the Centroid clusters.
        
    Cercanos : Int Array
        Array with the records more closed to the centroid.

    '''
    global df_cluster
    df_cluster = df_kmeans.copy()
    
    SSE = [] #return list
    for i in range(2,12):
        km = KMeans(n_clusters=i, random_state=0)
        km.fit(df_kmeans)
        SSE.append(km.inertia_)
    
    fig_sse = px.line(x=range(2,12), 
                    y=SSE, 
                    title='Elbow Method',
                    line_shape='spline',
                    labels={'x':'Number of Clusters K',
                            'y':'SSE'})
    fig_sse.update_traces(mode='markers+lines')

    kl = KneeLocator(range(2,12),SSE,curve='convex',direction='decreasing')
    fig_kl = px.line(x=range(2,12), 
                    y=SSE, 
                    title='Knee Point',
                    line_shape='spline',
                    labels={'x':'Number of Clusters K',
                            'y':'SSE'})
    fig_kl.update_traces(mode='markers+lines')
    fig_kl = fig_kl.add_vline(x= kl.elbow, line_width=4, 
                              fillcolor="red", opacity=0.6)
   
    
    MParticional = KMeans(n_clusters=kl.elbow, random_state=0).fit(df_kmeans)
    MParticional.predict(df_kmeans)
    
    
    df_cluster['clusterP'] = MParticional.labels_
    
    CentroidesP = MParticional.cluster_centers_
    l_col = [col for col in df_kmeans.columns.to_list() if col != 'clusterP']
    
    df_cluster_desc = pd.DataFrame(CentroidesP.round(kl.elbow), columns=l_col)
    df_cluster_desc['Cluster'] = [numCluster for numCluster in range(0,kl.elbow)]
    
    figure_3d = go.Figure()
    
    figure_3d.add_trace(go.Scatter3d(x=df_cluster.iloc[:,0], 
                                     y=df_cluster.iloc[:,1],
                                     z=df_cluster.iloc[:,2],
                                     mode='markers',
                                     marker = dict(
                                         size=6,
                                         symbol='circle',
                                         opacity=0.7,
                                         color = df_cluster['clusterP'].to_list()
                                         )
                                   )
                        )
    
    figure_3d.add_trace(go.Scatter3d(x=CentroidesP[:,0], 
                                     y=CentroidesP[:,1],
                                     z=CentroidesP[:,2],
                                     mode='markers',
                                     marker = dict(
                                         size=8,
                                         symbol='diamond',
                                         opacity=0.8,
                                         color = df_cluster['clusterP'].unique()
                                         )
                                   )
                        )
    
    Cercanos,_ = pairwise_distances_argmin_min(MParticional.cluster_centers_, 
                                               df_kmeans)
    
    return  fig_sse, fig_kl, df_cluster_desc, figure_3d, Cercanos


def logistic_regression_model(X, Y):
    '''
    Funtion to get a logistic regression model based on splitting data selected

    Parameters
    ----------
    X : Array
        Numpy Array with the X data, which are the variables for the linear
        model.
        
    Y : Array
        Numpy Array with the Y data, which are dependet variables for the linear
        model

    Returns
    -------
    score : Float
        Score of the Linear Model.
        
    conf_matrix : Pandas Cross Table
        Confusion Matrix or Classification Matrix
        
    exactitud : Float
        Float number which represent the pression of the model to predict values.
        
    report : Array
        Table with the classification report.
        
    intercept : Float
        Classification intercept value.
        
    coeffs : Float List
        Coefficient of the classification model variables.

    '''
    global final_model
    
    Clasificacion = linear_model.LogisticRegression() 
    seed = 1234
    X_train,X_validation,Y_train,Y_validation=model_selection.train_test_split(X, 
                                                                Y, 
                                                                test_size=0.2, 
                                                                random_state=seed, 
                                                                shuffle = True)
    Clasificacion.fit(X_train, Y_train)
    
    #Probabilidad = Clasificacion.predict_proba(X_train)
    #df_probabilidad = pd.DataFrame(Probabilidad)
    
    Predicciones = Clasificacion.predict(X_train)
    #df_predicciones = pd.DataFrame(Predicciones)
    
    score = Clasificacion.score(X_train, Y_train)
    PrediccionesNuevas = Clasificacion.predict(X_validation)
    conf_matrix = pd.crosstab(Y_validation.ravel(), PrediccionesNuevas, 
                                   rownames=['Real'], 
                                   colnames=['Clasificación'])
    
    iname = conf_matrix.index.name
    cname = conf_matrix.columns.name
    conf_matrix = conf_matrix.reset_index()
    conf_matrix.rename(columns={conf_matrix.columns[0]: iname + ' / ' + cname},
                       inplace=True)
    
    #Reporte de la clasificación
    exactitud = Clasificacion.score(X_validation, Y_validation)
    report = classification_report(Y_validation, PrediccionesNuevas,
                                   output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    #df_report = df_report.sort_values(by=['f1-score'], ascending=False)
    
    #Ecuación del modelo
    intercept =  Clasificacion.intercept_
    coeffs =  Clasificacion.coef_
    
    final_model = Clasificacion
    
    return score,conf_matrix,exactitud,df_report,intercept,coeffs


def build_banner():
    '''
    Funtion to build the top banner of the Application

    Returns
    -------
    Lits
        List of the HTMl and dash comonents to build this GUI section

    '''
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-text",
                children=[
                    html.H5("Data Mining APP"),
                    html.H6("Osmar Juárez"),
                ],
            ),
            html.Div(
                id="banner-logo",
                children=[
                    html.Button(
                        id="learn-more-button",children="Acerca de",n_clicks=0
                    ),
                    html.A(
                        html.Img(id="logo", src='assets/icon_analyze.gif',
                                 style={})
                    ),
                ],
            ),
        ],
    )


def build_tabs():
    '''
    Funtion to build the Tabs section displayed by the project

    Returns
    -------
    Lits
        Html and Dash componets to display the Tabs used to display the projects
        modules.

    '''
    return html.Div(
        id = "tabs",
        className = "tabs",
        children = [
            dcc.Tabs(
                id = "app-tabs",
                value = "tab1",
                className = "custom-tabs",
                children = [
                    dcc.Tab(
                        id = "file-selection-tab",#id = "Specs-tab",
                        label = "File Selection",
                        value = "tab1",
                        className = "custom-tab",
                        selected_className = "custom-tab--selected",
                    ),
                    dcc.Tab(
                        id = "eda-tab",#id = "Control-chart-tab", 
                        label = "Exploration Data Analysis",
                        value = "tab2",
                        className = "custom-tab",
                        selected_className = "custom-tab--selected",
                    ),
                    dcc.Tab(
                        id = "Feature-selection-tab", 
                        label = "Feature Selection",
                        value = "tab3",
                        className = "custom-tab",
                        selected_className = "custom-tab--selected",
                    ),
                    dcc.Tab(
                        id = "Clustering-tab", 
                        label = "Clustering",
                        value = "tab4",
                        className = "custom-tab",
                        selected_className = "custom-tab--selected",
                    ),
                    dcc.Tab(
                        id = "Logistic-regression-tab", 
                        label = "Logistic Regression",
                        value = "tab5",
                        className = "custom-tab",
                        selected_className = "custom-tab--selected",
                    ),
                ],
            )
        ],
    )


def build_eda_charts(df_nan,df_box,df_desc_num):
    '''
    Funciton to build the plots and tables for EDA dashboard.

    Parameters
    ----------
    df_nan : Pandas Dataframe
        Dataframe with the null and unique values per column.
        
    df_box : Pandas Dataframe
        Dataframe with the numeric columns for the dataframe to display as box plot.
        
    df_desc_num : Pandas Dataframe
        Dataframe with the describe dataframe function output.

    Returns
    -------
    list
        All the figures and html containers to display dashboard.

    '''
    return [html.Div(id='eda-dashboard',
                     children=[html.Br(),
                        html.Center([html.Label(" DATA DESCRIPTION ")]),
                        dash_table.DataTable(id='missing-null-val-tab',
                                             data = df_nan.to_dict('records'),
                                             columns=[{"name": i,"id": i} for i in df_nan.columns],
                                             style_header={
                                                 'backgroundColor': 'black',
                                                 'textAlign': 'center'
                                             },
                                             style_cell={
                                                     'backgroundColor':'#1e2130',
                                                     'color': 'white',
                                                     'padding': '10px',
                                                     'textAlign': 'center'
                                                 },
                                             ),
                        html.Br(),
                        dcc.Graph(id='box-plot',
                                  figure= px.box(df_box, 
                                                 title='Atypic Values Detection')),
                        html.Br(),
                        html.Center([html.Label(" DESCRIBING NUMERIC COLUMNS ")]),
                        html.Center([dash_table.DataTable(id='describe-num-tab',
                                             data = df_desc_num.to_dict('records'),
                                             columns=[{"name": i,"id": i} for i in df_desc_num.columns],
                                             style_header={
                                                 'backgroundColor': 'black',
                                                 'textAlign': 'center'
                                             },
                                             style_cell={
                                                     'backgroundColor':'#1e2130',
                                                     'color': 'white',
                                                     'padding': '10px',
                                                     'textAlign': 'center'
                                                 },
                                             )]),
                        html.Br(),
                        html.Br(),
                        ]
                     )
          ]
    

def build_feature_selection_charts():
    return [html.Div(id='feature-selection-dash',
                     children = [html.Br(),
             html.Div([
                    dcc.Dropdown(
                        id='xaxis',
                        options=[{'label': i, 'value': i} for i in df_n.columns],
                        value=df_n.columns.to_list()[0]
                    )
            ],style={'width': '100%', 'display': 'inline-block'}),
            html.Center([html.Div(id='scatter-plots')]),
            html.Br(),
            html.Center([dcc.Graph(id='correlation-matrix',
                      figure = px.imshow(df.corr(), width=900, height=900,
                                         title='Correlation Matrix')
                      )]),
            html.Br()
        ], style={'padding':10})
    ]


def build_clustering_selection():
    return [html.Div([
                       dcc.Dropdown(
                           id='dropdown-select-drop',
                           options=[{'label': i, 
                                     'value': i} for i in df_n.columns],
                           placeholder='Select the column to drop...'
                       )
               ],style={'width': '100%',
                        'display': 'inline-block'}),
                        html.Div(id='clustering-algorithm-charts'),
            ]


def build_clustering_charts():
    return [html.Div(id="clustering-dashbord",
                         children=build_clustering_selection()
                         )
            ]

def build_subtab_build_model():
    col_list = df_n.columns.to_list()   
    return [html.Div(id="build-model-subtab",
                     children=[html.Center([
                         html.Br(),
                         html.Label("Select Predictible Variables (X data)"),
                         html.Br(),
                         dcc.Checklist(id='x-data-checklist',
                                       options=[{'label':col,
                                                 'value':col} for col in col_list],
                                        labelStyle={'display': 'inline-block'}
                                        ),
                         html.Br(),
                         html.Label("Select Class Variable (Y data)"),
                         html.Br(),
                         dcc.Checklist(id='y-data-checklist',
                                       options=[{'label':col,
                                                 'value':col} for col in col_list],
                                        labelStyle={'display': 'inline-block'}
                                        ),
                         html.Br(),
                         html.Button("Submit", id="submit-split-data-btn",
                                           n_clicks = 0),
                         ]),
                         html.Br(),
                         html.Div(id='classifier-model-report-summary')    
                         ]
                     )
            ]


def multiple_input_data(column_name):
    #print("input-box-{}".format(column_name))
    return html.Div([
                    html.Label(column_name),
                    dcc.Input(id="input-box-{}".format(column_name), 
                            placeholder="Type value", 
                            type="text",
                            style={"margin-left": "18px"}),
                    html.Br(),
                    html.P(id='space1'),
                ]
            )
        


def build_subtab_predict_data():
    global input_state_id
    if model_vars.empty != True:
        l_col = model_vars.columns.to_list()
        input_box_children = [multiple_input_data(col) for col in l_col]
        return [html.Div(id='subtab-predict-data',
                         children = [html.Center([
                             html.Br(),
                             html.Label("Type new values for each variable in order to display a prediction of new data"),
                             html.Br(),
                             html.P(id='space2'),
                             html.Div(id='input-data-div',
                                      children = input_box_children),
                             html.Br(),
                             html.Button("Submit", id="submit-predict-data-btn",
                                         n_clicks = 0),
                             html.Br(),
                             html.Div(id='new-prediction-output')])
                          ]
                    )]
    else:
        return [html.Div(id="empty-dev-pd",
                         children=[dcc.Tab(id='wrn-msg-empty-data-pd')])]

                                   
                                         
def build_tab1_load_files():
    '''
    Function to build and display the dropdown selecton files and update button

    Returns
    -------
    list
        The html and dash components to display this selection fiel section.

    '''
    return [html.Div(id="select-menu-data",
                     children=[html.Br(),
                               dcc.Dropdown(
                                   id='dropdown-files',
                                   options= dd_select_dataset(),
                                   placeholder="Select dataset" 
                                   ),
                               html.Br(),
                               html.Center([html.Button("Update", id="value-data-set-btn",
                                           n_clicks = 0)]),
                               dcc.Store(id='memory-data-tab1'),
                               html.Br(),
                               html.P(id='space-load-data'),
                               dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select Files')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                },
                                multiple=True),
                               html.Div(id='output-data-upload'),

                               ]
                     )
        ]


def build_tab2_dash_eda():
    '''
    Function to Disyplay the EDA dashboard if the data has been selected.

    Returns
    -------
    list
        The html and dash companents to displayn or not the dashboard.

    '''
    if df.empty !=True:
        df_nan = missing_null_val()
        df_box = df[df.describe().columns.to_list()]
        df_describe_num = desc_data('number')
        return build_eda_charts(df_nan, df_box, df_describe_num)
    else:
        return [html.Div(id="empty-dev",
                         children=[dcc.Tab(id='wrn-msg-empty-data')])]


def build_tab3_dash_feature_sel():
    '''
    Funtion to build the tab 3 if the data was selected or not by the user

    Returns
    -------
    Lits
        Components html and dash to build the module feature selection.

    '''
    global df_n
    
    if df.empty !=True:
        df_n = df[df.describe().columns.to_list()] #getting only numeric col
        return build_feature_selection_charts()
    else:
        return [html.Div(id="empty-dev",
                         children=[dcc.Tab(id='wrn-msg-empty-data-2')])]


def build_tab4_clustering():
    '''
    Funtion to build the dashboard for claustering in case that a data has been 
    selected.

    Returns
    -------
    List
        Html and dash components to build in the UI the dashboard for clustering

    '''
    if df.empty !=True:
        return build_clustering_charts()
    else:
        return [html.Div(id="empty-dev-clustering",
                         children=[dcc.Tab(id='wrn-msg-empty-data-2')])]



def build_tab5_logisticReg():   
    if df.empty != True:
        return [html.Div(id = "tabs-logisticReg",
                    className = "tabs",
                    children = [
                            dcc.Tabs(
                            id = "app-subtabs",
                            value = "subtab1",
                            className = "custom-tabs",
                            children = [
                                dcc.Tab(
                                    id = "split-data-subtab",
                                    label = "Build Module",
                                    value = "subtab1",
                                    className = "custom-tab",
                                    selected_className = "custom-tab--selected",
                                ),
                                dcc.Tab(
                                    id = "final-user-predict-subtab",
                                    label = "Predict New Data",
                                    value = "subtab2",
                                    className = "custom-tab",
                                    selected_className = "custom-tab--selected",
                                )
                            ]
                            ),
                            html.Div(id="app-content-subtabs")
                        ]
                    )]
    else:
        return [html.Div(id="empty-dev-lr",
                         children=[dcc.Tab(id='wrn-msg-empty-data-lr')])]
    
                
    

#building the front end app
app.layout = html.Div(id="big-app-container",
                      children=[build_banner(),
                                html.Div(
                                    id="app-container",
                                    children=[
                                        build_tabs(),
                                        html.Div(id="app-content")
                                        ],
                                    ),
                                ],
                      )


@app.callback(
    [Output("new-prediction-output","children")],
    [Input("submit-predict-data-btn","n_clicks")],
    #[State("input-box-{}".format(_),"value") for _ in tuple(model_vars.columns.to_list())]
)
def update_prediction_data_subtab(submit_click_sub2):#,*vals):
    if submit_click_sub2 != 0:
        df_newData = pd.DataFrame(columns = model_vars.columns.to_list())
        df_prediction = pd.DataFrame(columns = predict_vars.columns.to_list())
         
        #24.54	181.0	0.05263	0.04362	0.1587	0.05884 
        newData = [12067, 909,892, 500, 10000,321204,1]        
        
        newDataSerie = pd.Series(newData, index=model_vars.columns.to_list())
    
        df_newData = df_newData.append(newDataSerie, ignore_index=True)
        
        
        val_predict = final_model.predict(df_newData)
        predictSerie = pd.Series([val_predict], index=predict_vars.columns.to_list())
        
        df_prediction = df_prediction.append(predictSerie, ignore_index=True)
        
        return [html.Div(id="msg-predict",
                         children=[html.Br(),
                                   html.P(id='space3'),
                                   html.Label("Prediction Data"),
                                   html.Br(),
                                   dash_table.DataTable(id='prediction-output-table',
                                             data = df_prediction.to_dict('records'),
                                             columns=[{"name": i,"id": i} for i in df_prediction.columns],
                                             style_header={
                                                 'backgroundColor': 'black',
                                                 'textAlign': 'center'
                                             },
                                             style_cell={
                                                     'backgroundColor':'#1e2130',
                                                     'color': 'white',
                                                     'padding': '10px',
                                                     'textAlign': 'center'
                                                 },
                                             ),
                                   ])]
    else:
        return [html.Div(id="empty-dev-predit",
                         children=[dcc.Tab(id='wrn-msg-empty-data-predict')])]


@app.callback(  
    [Output('classifier-model-report-summary', "children")],
    [Input("submit-split-data-btn", "n_clicks")],
    [State("x-data-checklist","value"),
     State("y-data-checklist","value")]
)
def update_split_data(submit_click, x_data, y_data):
    if submit_click != 0 and len(x_data)!=0 and len(y_data)!=0:
        global model_vars, predict_vars
        
        X = np.array(df[x_data])
        Y = np.array(df[y_data])
        
        model_vars = df[x_data]
        predict_vars = df[y_data]
        
        score,conf_matrix,exactitud,report,intercept,coeffs = logistic_regression_model(X,Y)  
        return [html.Div(id="classifier-model-output",
                         children=[html.Br(),
                                   html.Center([html.Label("SCORE MODEL  >>  {}".format(score))]),
                                   html.Br(),
                                   html.Center([html.Label("ACCURACY MODEL  >>  {}".format(exactitud))]),
                                   html.Br(),
                                   html.Center([html.Label("CLASIFICATION MATRIX")]),
                                   dash_table.DataTable(id='conf-matrix',
                                             data = conf_matrix.to_dict('records'),
                                             columns=[{"name": i,"id": i} for i in conf_matrix.columns],
                                             style_header={
                                                 'backgroundColor': 'black',
                                                 'textAlign': 'center'
                                             },
                                             style_cell={
                                                     'backgroundColor':'#1e2130',
                                                     'color': 'white',
                                                     'padding': '10px',
                                                     'textAlign': 'center'
                                                 },
                                             ),
                                   html.Br(),
                                   html.Center([html.Label("CLASIFICATION REPORT")]),
                                   dash_table.DataTable(id='classification-report',
                                             data = report.to_dict('records'),
                                             columns=[{"name": i,"id": i} for i in report.columns],
                                             style_header={
                                                 'backgroundColor': 'black',
                                                 'textAlign': 'center'
                                             },
                                             style_cell={
                                                     'backgroundColor':'#1e2130',
                                                     'color': 'white',
                                                     'padding': '10px',
                                                     'textAlign': 'center'
                                                 },
                                             ),
                                   html.Br()
                             ]
                         )
                ]
    else:
        return [html.Div(id="empty-dev-lr",
                         children=[dcc.Tab(id='wrn-msg-empty-data-lr')])]


@app.callback(  
    [Output("app-content-subtabs", "children")],
    [Input("app-subtabs", "value")],
)
def render_subtabs(subtab_switch):
    '''
    Function to update the subTabs selection for the logistic regresssion module.

    Parameters
    ----------
    tab_switch : Str
        ID for the dash component of the subtab selected by the user.

    Returns
    -------
    List
        Based on the subtab slected by the user a function returns building the 
        dash and html components for the substab selected.

    '''
    if subtab_switch == 'subtab1':
        return build_subtab_build_model()
    elif subtab_switch == 'subtab2':
        return build_subtab_predict_data()



@app.callback(
    Output('clustering-dashbord','children'),
    [Input('dropdown-select-drop','value')])
def update_df_column_dropped(column_to_drop):
    '''
    Update the clustering dashboard based on the column dropped to predict
    Parameters
    ----------
    column_to_drop : Str
        Column to drop from the dataframe to predict values.

    Returns
    -------
    List
        Html and dash components with the clustering algorithm plots.

    '''
    if column_to_drop != None:
        global df_kmeans
        df_kmeans = df_n.copy()
        #print(df.columns)
        df_kmeans = df_n.drop([column_to_drop], axis=1)
        #print(df_kmeans.columns)
        out1_sse,out2_kl,out3_descCluster,out4_fig,out5_cercanos = kmeans_algorithm()
        
        df_grpby_cluster = df_cluster.groupby('clusterP',
                                              as_index=False).count()
        df_grpby_cluster = df_grpby_cluster.iloc[:,0:2]
        df_grpby_cluster.columns = ['Clusters','Number of Records']
        
        df_nearCent = df_cluster.iloc[out5_cercanos]
        
        return [html.Div(id="kmeans-algorithm-charts",
                             children=[dcc.Graph(id='elbow-method-chart',
                                                 figure= out1_sse),
                                       html.Br(),
                                       dcc.Graph(id='knee-point-chart',
                                                 figure= out2_kl),
                                       html.Br(),
                                       html.Center([html.Label(" DESCRIBING RESULTS OF CLUSTERING PROCESS ")]),
                                       html.Br(),
                                       dash_table.DataTable(id='num-clusters',
                                             data = df_grpby_cluster.to_dict('records'),
                                             columns=[{"name": i,"id": i} for i in df_grpby_cluster.columns],
                                             style_header={
                                                 'backgroundColor': 'black',
                                                 'textAlign': 'center'
                                             },
                                             style_cell={
                                                     'backgroundColor':'#1e2130',
                                                     'color': 'white',
                                                     'padding': '10px',
                                                     'textAlign': 'center'
                                                 },
                                             ),
                                       html.Br(),
                                       dcc.Graph(id='scatter-clusters-plot',
                                         figure=px.scatter(df_cluster,
                                                        x=df_cluster.iloc[:,0], 
                                                        y=df_cluster.iloc[:,1], 
                                                        color='clusterP')
                                         ),
                                       html.Br(),
                                       dash_table.DataTable(id='describing-clusters-table',
                                             data = out3_descCluster.to_dict('records'),
                                             columns=[{"name": i,"id": i} for i in out3_descCluster.columns],
                                             style_header={
                                                 'backgroundColor': 'black',
                                                 'textAlign': 'center'
                                             },
                                             style_cell={
                                                     'backgroundColor':'#1e2130',
                                                     'color': 'white',
                                                     'padding': '10px',
                                                     'textAlign': 'center'
                                                 },
                                             ),
                                       html.Br(),
                                       dcc.Graph(id='3d-centroid-plot',
                                                 figure = out4_fig),
                                       html.Br(),
                                       html.Center([html.Label("Near data to the centroids")]),
                                       dash_table.DataTable(id='near-data-centroid-table',
                                             data = df_nearCent.to_dict('records'),
                                             columns=[{"name": i,"id": i} for i in df_nearCent.columns],
                                             style_header={
                                                 'backgroundColor': 'black',
                                                 'textAlign': 'center'
                                             },
                                             style_cell={
                                                     'backgroundColor':'#1e2130',
                                                     'color': 'white',
                                                     'padding': '10px',
                                                     'textAlign': 'center'
                                                 },
                                             ),
                                       html.Br()
                                       ]
                             )
                ]
    else:
        return build_clustering_selection()

  
@app.callback(
    Output('scatter-plots','children'),
    [Input('xaxis', 'value')])
def update_feature_scatter_graph(xaxis_name):
    '''
    Update the scatter graphs based on the column selected

    Parameters
    ----------
    xaxis_name : Str
        Column selected to build the scatter plots.

    Returns
    -------
    List
        Html and dash components with the scatter plots to build in the dash app

    '''
    return multiple_scatter_graphs(xaxis_name)


@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def update_upload_file(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(
    [Output("memory-data-tab1", "data")],
    [Input("value-data-set-btn","n_clicks"),
     Input("dropdown-files", "value")],    
)
def set_data_selection(clicks,drop_down_value):
    '''
    Function to update the data selection in the GUI and the whole APP

    Parameters
    ----------
    clicks : Int
        Number of clicks to the Update button for the file selection tab.
    
    drop_down_value : Str
        File selected by the user for the list of files previiuos loaded .

    Returns
    -------
    data_pth : Lits
        List with the path for the file selected by the user.

    '''
    global df 
    
    if clicks != 0:
        data_pth = [drop_down_value]
        df = read_file(data_pth[0])
    else:
        data_pth = [None]
    return data_pth


@app.callback(  
    [Output("app-content", "children")],
    [Input("app-tabs", "value")],
)
def render_tab_content(tab_switch):
    '''
    Function to update the Tab selection for the APP.

    Parameters
    ----------
    tab_switch : Str
        ID for the dash component of the tab selected by the user.

    Returns
    -------
    List
        Based on the tab slected by the user a function returns building the 
        dash and html components for the tab selected.

    '''
    if tab_switch == 'tab1':
        return build_tab1_load_files()
    elif tab_switch == 'tab2':
        return build_tab2_dash_eda()
    elif tab_switch == 'tab3':
        return build_tab3_dash_feature_sel()
    elif tab_switch == 'tab4':
        return build_tab4_clustering()
    elif tab_switch == 'tab5':
        return build_tab5_logisticReg()
    
    
    
    
if __name__ == '__main__':
    app.run_server()

 
#TODO: in feature selection update figure scatter points with different colors 