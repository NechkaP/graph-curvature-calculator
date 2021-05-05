import dash
from dash.dependencies import Input, Output, State
import json

import pandas as pd
import base64
import datetime
import io
import time
from colour import Color
from datetime import datetime
import json

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

import plotly.graph_objs as go
from index import *
from graph import *
from Math import *


G = Graph()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_scripts = ["https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"]
app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                external_scripts=external_scripts)
server = app.server
app.title = "Curvature Calculator"


app.layout = LAYOUT


@app.callback(
    Output('hover-data', 'children'),
    [Input('my-graph', 'hoverData')],
    [State('my-graph', 'figure')])
def display_hover_data(hoverData, figure):
    if hoverData != None:
        curve_number = hoverData['points'][0]['curveNumber']
        trace_name = figure['data'][curve_number]['name']
        return trace_name
    return json.dumps(hoverData)

    
@app.callback(
    Output('my-graph', 'figure'),
    [Input('my-graph', 'selectedData'),
    Input('layout-type', 'value'),
    Input('colortheme', 'value'),
    Input('matrix-input', 'value'),
    Input('curvature-type', 'value'),
    Input('edge-button', 'n_clicks'),
    Input('edge-delete-button', 'n_clicks'),
    Input('vertex-button', 'n_clicks'),
    Input('vertex-delete-button', 'n_clicks'),
    Input('upload-data', 'contents')],
    [State('my-graph', 'figure'),
    State('upload-data', 'filename')])

def change_layout(clickData, layout_type, color, matrix, curvature_type, n_clicks, n_clicks_delete, n_clicks_v,
                  n_clicks_v_delete, contents, 
                  figure, filename):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    global G
    
    if 'colortheme' in changed_id:
        if color != None and 'D' in color:
            G.dark = True
        else:
            G.dark = False
    
    elif 'my-graph' in changed_id:
        G.trace_recode_init()
        G.selected = []
        for n in G.nodes:
            n.selected = False
        if clickData and clickData['points']:
            for item in clickData['points']:
                curve_number = item['curveNumber']
                name = figure['data'][curve_number]['name']
                if name != 'HEATMAP': #name != '' and name != 'HEATMAP':
                    G.selected.append(name)
                    try:
                        G.nodes[int(name)].selected = True
                    except:
                        pass
                elif name == 'HEATMAP':
                    G.selected = []
                    break
            
    elif 'edge-button' in changed_id:
        if len(G.selected) >= 2:
            from_ = int(G.selected[-2])
            to = int(G.selected[-1])
            G.add_edge(from_, to)
            G.add_edge(to, from_)
            
    elif 'edge-delete-button' in changed_id:
        if len(G.selected) >= 2:
            from_ = int(G.selected[-2])
            to = int(G.selected[-1])
            G.delete_edge(from_, to)
            G.delete_edge(to, from_)
            
    elif 'vertex-button' in changed_id:
        G.add_node()
        
    elif 'vertex-delete-button' in changed_id:
        index = int(SELECTED[-1])
        G.delete_node(index)
    
    elif 'matrix-input' in changed_id:
        nums = list(map(float, matrix.replace(']', '').replace('[', '').replace(',', ' ').split()))
        N = int(np.sqrt(len(nums)))
        G.M = np.zeros(shape = (N, N), dtype = float)
        for i in range(N):
            G.add_node(node_id=i)
        for i in range(N):
            for j in range(N):
                G.M[i][j] = nums[i * N + j]
                G.add_edge(from_node=i, to_node=j)
                G.add_edge(from_node=j, to_node=i)
                

    elif 'upload-data' in changed_id:
        contents = contents[0]
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        dat = decoded.decode('utf-8')
        nums = list(map(float, dat.replace(']', '').replace('[', '').replace(',', ' ').split()))
        N = int(np.sqrt(len(nums)))
        G.M = np.zeros(shape = (N, N), dtype = float)
        for i in range(N):
            for j in range(N):
                G.M[i][j] = nums[i * N + j]
                
                #if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            #df = pd.read_csv(
            #    io.StringIO(decoded.decode('utf-8')))
    
    #K = curvature(M, p, curvature_type)  
    return G.draw(layout=layout_type)


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)