#app.py

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

class LastClick:
    def __init__(self):
        self.x = None
        self.y = None

last_click = LastClick()
idleness = None

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_scripts = ["https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"]
app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                external_scripts=external_scripts)
server = app.server
app.title = "Curvature Calculator"


app.layout = LAYOUT


@app.callback(
    Output('clear-all-button', 'children'),
    Input('confirm-delete-provider', 'submit_n_clicks')
)
def confirm_delete(submit_n_clicks):
    if not submit_n_clicks:
        return ''
    return submit_n_clicks
    
    
@app.callback(
    Output('my-graph', 'figure'),
    [Input('my-graph', 'selectedData'),
    Input('layout-type', 'value'),
    Input('colortheme', 'value'),
    Input('matrix-input', 'value'),
    Input('curvature-type', 'value'),
    Input('idleness', 'value'),
    Input('edge-button', 'n_clicks'),
    Input('edge-weight', 'value'),
    Input('edge-delete-button', 'n_clicks'),
    Input('vertex-button', 'n_clicks'),
    Input('vertex-delete-button', 'n_clicks'),
    Input('clear-all-button', 'children'),
    Input('upload-data', 'contents'),
    Input('weighted-mode', 'value')],
    [State('my-graph', 'figure'),
    State('upload-data', 'filename')])
def change_layout(clickData,
                  layout_type,
                  color,
                  matrix,
                  curvature_type,
                  idleness_input,
                  n_clicks,
                  edge_weight, 
                  n_clicks_delete,
                  n_clicks_v,
                  n_clicks_v_delete,
                  n_clicks_clear,
                  contents,
                  weighted_mode,  
                  figure, 
                  filename):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    global G
    weighted = (weighted_mode == 'weighted')
    if 'idleness' in changed_id:
        global idleness
        if idleness_input:
            try:
                idleness = float(idleness_input)
                if idleness < 0.0 or idleness > 1.0:
                    idleness = None
            except:
                idleness = None
        else:
            idleness = None
        
    if 'layout-type' in changed_id:
        return G.draw(layout=layout_type, curvature_type=curvature_type, fixed_pos=False, weighted=weighted, idleness=idleness)
    
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
                    if len(G.selected) > 2:
                        G.selected = G.selected[-2:]
                elif name == 'HEATMAP':
                    G.selected = []
                    break
        for name in G.selected:
            G.nodes[int(name)].selected = True
            
    elif 'edge-button' in changed_id:
        weight = 1.0
        if edge_weight:
            try:
                weight = float(edge_weight)
                if weight < 0.0:
                    weight = 1.0
            except:
                pass
        if len(G.selected) >= 2:
            from_ = int(G.selected[-2])
            to = int(G.selected[-1])
            G.add_edge(from_, to, weight)
            
    elif 'edge-delete-button' in changed_id:
        if len(G.selected) >= 2:
            from_ = int(G.selected[-2])
            to = int(G.selected[-1])
            G.delete_edge(from_, to)
            
    elif 'vertex-button' in changed_id:
        if last_click.x:
            G.add_node(x=last_click.x, y=last_click.y)
            last_click.x = None
            last_click.y = None
        else:
            G.add_node(x=last_click.x, y=last_click.y)
            return G.draw(layout=layout_type, curvature_type=curvature_type, fixed_pos=False, weighted=weighted, idleness=idleness)
        
    elif 'vertex-delete-button' in changed_id:
        if len(G.selected) > 0:
            index = int(G.selected[-1])
            G.delete_node(index)
    
    elif 'clear-all-button' in changed_id:
        G.nodes.clear()
        G.edges.clear()
        G = Graph(nodes=[], edges=[], dark=G.dark)
    
    elif 'matrix-input' in changed_id:
        G.nodes.clear()
        G.edges.clear()
        nums = list(map(float, matrix.replace(']', '').replace('[', '').replace(',', ' ').split()))
        N = int(np.sqrt(len(nums)))
        G.M = np.zeros(shape = (N, N), dtype = float)
        G.DM = np.zeros(shape = (N, N), dtype = float)
        for i in range(N):
            G.add_node(node_id=i, resize=False)
        for i in range(N):
            for j in range(N):
                if nums[i * N + j]:
                    G.add_edge(from_node=i, to_node=j)
                

    elif 'upload-data' in changed_id:
        contents = contents[0]
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        dat = decoded.decode('utf-8')
        nums = list(map(float, dat.replace(']', '').replace('[', '').replace(',', ' ').split()))
        N = int(np.sqrt(len(nums)))
        G.M = np.zeros(shape=(N, N), dtype=float)
        G.DM = np.zeros(shape = (N, N), dtype = float)
        for i in range(N):
            for j in range(N):
                G.M[i][j] = nums[i * N + j]
                G.DM[i][j] = nums[i * N + j]
                
                #if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            #df = pd.read_csv(
            #    io.StringIO(decoded.decode('utf-8')))
    if curvature_type == 'directed':
        return G.draw(layout=layout_type, curvature_type=curvature_type, directed=True, weighted=weighted)
    
    return G.draw(layout=layout_type, curvature_type=curvature_type, weighted=weighted, idleness=idleness)


#@app.callback(
#    Output('colortheme', 'value'),
#    Input('my-graph', 'clickData'),
#    [State('colortheme', 'value'),
#    State('my-graph', 'figure')]
#    )
#def click_processing(clickData, colorth, figure):
#    if clickData is None or clickData['points'] == []:
#        return colorth
#    curve = clickData['points'][0]['curveNumber']
#    name = figure['data'][curve]['name']
#    if name == 'HEATMAP':
#        global last_click
#        last_click.x = int(clickData['points'][0]['x'])
#        last_click.y = int(clickData['points'][0]['y'])
#        global G
#        G.selected = []
#    return colorth


@app.callback(
    Output('matrix-input', 'value'),
    Input('clear-all-button', 'children')
)
def clear_input(n_clicks):
    return ''


@app.callback(
    Output('edge-weight', 'value'),
    Input('clear-all-button', 'children')
)
def clear_input2(n_clicks):
    return ''


@app.callback(
    Output('warnings-area', 'value'),
    [Input('edge-weight', 'value'),
    Input('idleness', 'value'),
    Input('matrix-input', 'value')]
)
def validate_and_display_warning(edge_weight, idleness_input, matrix):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'idleness' in changed_id:
        #global idleness
        if idleness_input:
            try:
                idleness = float(idleness_input)
                if idleness < 0.0 or idleness > 1.0:
                    idleness = None
                    return 'Idleness parameter should be a real number belonging to interval [0.0, 1.0]. Computing curvature with idleness p(x)=1/(d(x)+1)...'
            except:
                idleness = None
                return 'Idleness parameter should be a real number belonging to interval [0.0, 1.0]. Computing curvature with idleness p(x)=1/(d(x)+1)...'
        else:
            idleness = None
            return 'Idleness parameter should be a real number belonging to interval [0.0, 1.0]. Computing curvature with idleness p(x)=1/(d(x)+1)...'
            
    elif 'weight' in changed_id:
        if edge_weight:
                try:
                    weight = float(edge_weight)
                    if weight < 0.0:
                        return 'Edge weight should be a real non-negative number. Setting the next edge weight to 1.0...'
                except:
                    pass
    
    return ''


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
