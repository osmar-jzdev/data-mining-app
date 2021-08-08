'''
@author: Osmar Jrz

This app explore data from a file and export the results into a dashboard
to understand the key results from the exploration data process.

This process it is the first step for a Data Mining Project
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


#global variables
df = pd.DataFrame() #empty dataframe

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
                        id="learn-more-button", children="Acerca de", n_clicks=0
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
                        id = "eda-tab",#id = "Control-chart-tab", # id = "upload-file-tab"
                        label = "Exploration Data Analysis",
                        value = "tab2",
                        className = "custom-tab",
                        selected_className = "custom-tab--selected",
                    ),
                    dcc.Tab(
                        id = "Feature-selection-tab", # id = "upload-file-tab"
                        label = "Feature Selection",
                        value = "tab3",
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
                                                            'backgroundColor': '#1e2130',
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
                                                            'backgroundColor': '#1e2130',
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
    

def build_feature_scatter_graph(df_n):
    return [html.Div(id='feature-selection-dash',
                     children = [html.Br(),
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='xaxis',
                        options=[{'label': i, 'value': i} for i in df_n.columns],
                        value=df_n.columns.to_list()[0]
                    )
                ],
                style={'width': '48%', 'display': 'inline-block'}),
         
                html.Div([
                    dcc.Dropdown(
                        id='yaxis',
                        options=[{'label': i, 'value': i} for i in df_n.columns],
                        value=df_n.columns.to_list()[1]
                    )
                ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ]),
            dcc.Graph(id='feature-graphic'),
            html.Br(),
            html.Center([dcc.Graph(id='correlation-matrix',
                      figure = px.imshow(df.corr(), width=900, height=900)
                      )]),
            html.Br()
        ], style={'padding':10})
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
    if df.empty !=True:
        df_n = df[df.describe().columns.to_list()] #getting only numeric col
        return build_feature_scatter_graph(df_n)
    else:
        return [html.Div(id="empty-dev",
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
    Output('feature-graphic', 'figure'),
    [Input('xaxis', 'value'),
     Input('yaxis', 'value')])
def update_feature_scatter_graph(xaxis_name, yaxis_name):
    return {
        'data': [go.Scatter(
            x=df[xaxis_name],
            y=df[yaxis_name],
            #text=df['name'],
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            }
        )],
        'layout': go.Layout(
            xaxis={'title': xaxis_name},
            yaxis={'title': yaxis_name},
            #margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }



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
