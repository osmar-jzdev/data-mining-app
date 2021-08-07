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
                        id = "Claustering-tab", # id = "upload-file-tab"
                        label = "Claustering",
                        value = "tab3",
                        className = "custom-tab",
                        selected_className = "custom-tab--selected",
                    ),
                ],
            )
        ],
    )


def read_file(file):
    if '.csv' in str(file):
        df = pd.read_csv(file)
    elif '.xlsx' in str(file):
        df = pd.read_excel(file)
    elif '.txt' in str(file):
        df = pd.read_table(file)
    
    return df


#def start_dm():
    #df = px.data.tips()
    #df = read_file('data/Hipoteca.csv')
#    fig = px.histogram(df, x="total_bill", nbins=20)
#    fig.show()    
#    print("")


def dd_select_dataset():
    app_path = str(pathlib.Path(__file__).parent.resolve())
    data_path = app_path + '\\data'
    
    file_list = list()
    for file in os.listdir(data_path):
        file_list.append({'label':file, 'value':data_path + '\\' + file})
                 
    return file_list


def missing_null_val():
    #obteniendo total de valores nulos en el dataframe
    df_summary = pd.DataFrame(df.isna().sum(), columns = ['Count Null Values'])
    df_summary = pd.DataFrame(df.isna().sum(), columns = ['Count Null Values'])
    #obteniendo valores unicos y total de registros por cada columna
    df_summary['Nunique values'] = df.nunique().to_list()
    df_summary['Count Records'] = df.count().to_list()
    df_summary['Columns'] = df_summary.index.to_list()
    
    #ordenando las columnas
    df_summary = df_summary[['Columns','Count Records',
                             'Count Null Values','Nunique values']]
    #print(df_summary)

    return df_summary


def build_eda_charts(df_nan,df_box):
    return [html.Div(id='eda-dashboard',
                     children=[html.Br(),
                               #dcc.Graph(id='descripting-table2',
                               #  figure={
                               #      'data': [],
                               #      'layout': go.Layout()
                               #      }
                               #  ),
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
                                         figure= px.box(df_box)),
                               html.Br()
                               ]
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
        df_nan = missing_null_val()
        df_box = df[df.describe().columns.to_list()]
        #empty = pd.DataFrame()
        #print(df_nan.to_dict())
        return build_eda_charts(df_nan, df_box)
    else:
        return [html.Div(id="empty-dev",
                         children=[dcc.Tab(id='wrn-msg-empty-data')])]


def build_tab3_dash_claustering():
    #if df.empty !=True:
        #df_nan = missing_null_val()
        #empty = pd.DataFrame()
        #print(df_nan.to_dict())
        #return build_claustering_charts(df_nan)
    #else:
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

  

@app.callback(
    [Output("memory-data-tab1", "data")],
    [Input("value-data-set-btn","n_clicks"),
     Input("dropdown-files", "value")],    
)
def set_data_selection(clicks,drop_down_value):
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
    if tab_switch == 'tab1':
        return build_tab1_dropdown_files()
    elif tab_switch == 'tab2':
        return build_tab2_dash_eda()
    elif tab_switch == 'tab3':
        return build_tab3_dash_claustering()



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
