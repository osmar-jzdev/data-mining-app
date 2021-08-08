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
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
#manage os system and paths
import os
import pathlib
#to use algorithm K Means with elbow method 
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D

#global variables
df = pd.DataFrame() #empty dataframe with the original data loaded
df_kmeans = pd.DataFrame() #empty dataframe for kmeans algorithm 
df_cluster = pd.DataFrame() #empty dataframe for kmeans output - clustering out

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
    
    df_summary = df_summary[['Columns','Count Records',
                             'Count Null Values','Nunique values']]
    #print(df_summary)

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
    df_n = df[df.describe().columns.to_list()]
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
    global df_cluster
    df_cluster = df_kmeans.copy()
    
    SSE = [] #return list
    for i in range(2,12):
        km = KMeans(n_clusters=i, random_state=0)
        km.fit(df_kmeans)
        SSE.append(km.inertia_)
        
    kl = KneeLocator(range(2,12),SSE,curve='convex',direction='decreasing')
    kl.elbow
    #return kl
    print(type(kl))
    MParticional = KMeans(n_clusters=4, random_state=0).fit(df_kmeans)
    MParticional.predict(df_kmeans)
    MParticional.labels_
    
    df_cluster['clusterP'] = MParticional.labels_
    
    CentroidesP = MParticional.cluster_centers_
    l_col = [col for col in df_kmeans.columns.to_list() if col != 'clusterP']
    df_cluster_desc = pd.DataFrame(CentroidesP.round(4), columns=l_col)
    
    '''
    plt.rcParams['figure.figsize'] = (10, 7)
    plt.style.use('ggplot')
    colores=['red', 'blue', 'green', 'yellow']
    asignar=[]
    for row in MParticional.labels_:
        asignar.append(colores[row])
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(MHipoteca.iloc[:, 0], MHipoteca.iloc[:, 1], MHipoteca.iloc[:, 2], marker='o', c=asignar, s=60)
    ax.scatter(CentroidesP[:, 0], CentroidesP[:, 1], CentroidesP[:, 2], marker='*', c=colores, s=1000)
    plt.show()
    '''
    
    figure_3d = ''
    Cercanos,_ = pairwise_distances_argmin_min(MParticional.cluster_centers_, df_kmeans)
    #Cercanos
    #plt.style.use('ggplot')
    #kl.plot_knee()
    
    
    return  SSE, df_cluster_desc, figure_3d, Cercanos

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
                        html.Label(" Exploration columns "),
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
                        html.Label(" Describing numeric columns "),
                        dash_table.DataTable(id='describe-num-tab',
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
                                             ),
                        html.Br(),
                        html.Br(),
                        ]
                     )
          ]
    

def build_feature_selection_charts(df_n):
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
                                     'value': i} for i in df.columns],
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

                                       
                                         
def build_tab1_dropdown_files():
    '''
    Function to build and display the dropdown selecton files and update button

    Returns
    -------
    list
        The html, adn dash components to display this selection fiel section.

    '''
    return [html.Div(id="select-menu-data",
                     children=[html.Br(),
                               dcc.Dropdown(
                                   id='dropdown-files',
                                   options= dd_select_dataset(),
                                   placeholder="Select dataset" 
                                   ),
                               html.Br(),
                               html.Button("Update", id="value-data-set-btn",
                                           n_clicks = 0),
                               dcc.Store(id='memory-data-tab1')
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
    if df.empty !=True:
        df_n = df[df.describe().columns.to_list()] #getting only numeric col
        return build_feature_selection_charts(df_n)
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
    #Output('clustering-algorithm-charts','children'),
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
    print("\nColumn to drop: ", column_to_drop)
    if column_to_drop != None:
        global df_kmeans
        df_kmeans = df.copy()
        print(df.columns)
        df_kmeans = df.drop([column_to_drop], axis=1)
        print(df_kmeans.columns)
        out1_sse, out2_df_desc_cluster, out3_fig, out4_cercanos = kmeans_algorithm()
        #df_cluster.groupby('clusterP')['clusterP'].count()#table 
        
        #df_cluster scatter plot
        #plt.figure(figsize=(8,4))
        #plt.scatter(MHipoteca['ingresos'], MHipoteca['gastos_comunes'], c=MParticional.labels_, cmap='rainbow')
        #plt.show()
        return [html.Div(id="empty-dev-clustering",
                             children=[dcc.Tab(id='wrn-msg-empty-data-2')])]
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
        return build_tab1_dropdown_files()
    elif tab_switch == 'tab2':
        return build_tab2_dash_eda()
    elif tab_switch == 'tab3':
        return build_tab3_dash_feature_sel()
    elif tab_switch == 'tab4':
        return build_tab4_clustering()
    



if __name__ == '__main__':
    app.run_server()
    


#@app.callback(
#    [Output('missing-null-val-tab', 'data')],
#    [Input('missing-null-val-tab', 'data')],
#)
#def update_missing_null_val_tab(data):
#    df = missing_null_val()
#    data = df.to_dict('records')
#    return data
