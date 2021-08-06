'''
@author: Osmar Jrz

This app explore data from a file and export the results into a dashboard
to understand the key results from the exploration data process.

This process it is the first step for a Data Mining Project
'''
# for manage data 
import pandas as pd 
import numpy as np
# for plot
import seaborn as sns 
#for web dashboard 
import dash 
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
#manage os system and paths
import os
import pathlib


#global variables
df = pd.DataFrame() #empty dataframe

app  = dash.Dash()
app.title = "Data Mining APP"



def build_banner():
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
                        id = "Specs-tab",#id = "file-selection-tab"
                        label = "File Selection",
                        value = "tab1",
                        className = "custom-tab",
                        selected_className = "custom-tab--selected",
                    ),
                    dcc.Tab(
                        id = "Control-chart-tab", # id = "upload-file-tab"
                        label = "Exploration Data Analysis",
                        value = "tab2",
                        className = "custom-tab",
                        selected_className = "custom-tab--selected",
                    ),
                ],
            )
        ],
    )


def read_file(file, flagType):
    if flagType == '.csv':
        df = pd.read_csv(file)
    elif flagType == '.xlsx':
        df = pd.read_excel(file)
    
    return df


def start_dm():
    
    #df = px.data.tips()
    df = read_file('data/Hipoteca.csv', '.csv')
    fig = px.histogram(df, x="total_bill", nbins=20)
    fig.show()    
    print("")


def dd_select_dataset():
    app_path = str(pathlib.Path(__file__).parent.resolve())
    data_path = app_path + '\\data'
    
    file_list = list()
    for file in os.listdir(data_path):
        file_list.append({'label':file, 'value':data_path + '\\' + file})
                 
    return file_list


def build_eda_charts():
    print(df.head())

    return [html.Div(id='eda-div1',
                     children=[html.Br(),
                               dcc.Graph(id='descripting-table1',
                                figure={
                                    'data': [],
                                    'layout': go.Layout()
                                    }
                                ),
                         html.Br(),
                         html.Div(id='eda-div2',
                                 children=[html.Br(),
                                           dcc.Graph(id='descripting-table2',
                                            figure={
                                                'data': [],
                                                'layout': go.Layout()
                                                }
                                            )
                                     ]
                                 )]
                     )
          ]
    
    


def build_tab1_dropdown_files():
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
    if df.empty !=True:
        return build_eda_charts()
    else:
        return [html.Div(id="empty-dev",
                         children=[dcc.Tab(id='wrn-msg-empty-data')])]
                     

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

'''
@app.callback(
    [Output("memory-data-eda", "data")],
    [Input("memory-data-tab1", "data")],
)
def update_tab_charts_modules(data_selection):
    global data 
    
    if data_selection.len()!=0:
        data = read_file(data[0], '.csv')
        
    return [data]
'''   

@app.callback(
    [Output("memory-data-tab1", "data")],
    [Input("value-data-set-btn","n_clicks"),
     Input("dropdown-files", "value")],    
)
def set_data_selection(clicks,drop_down_value):
    global df 
    
    if clicks != 0:
        data_pth = [drop_down_value]
        df = read_file(data_pth[0], '.csv')
    else:
        data_pth = [None]
    return data_pth


@app.callback(  
    [Output("app-content", "children")],
    [Input("app-tabs", "value")],
)
def render_tab_content(tab_switch):
    if tab_switch == 'tab1':
        return build_tab1_dropdown_files()
    elif tab_switch == 'tab2':
        return build_tab2_dash_eda()



if __name__ == '__main__':
    #start_dm()
    app.run_server()
    
