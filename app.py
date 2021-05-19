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
from graph import *
from Math import *
from index import *
from serialize import *


class LastClick:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        
    def serialize(self):
        return json.dumps({'x' : self.x, 'y' : self.y})
    
    def deserialize(self, s):
        data = json.loads(s)
        print(data)
        self.x = data['x']
        self.y = data['y']
    

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_scripts = ["https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"]
app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                external_scripts=external_scripts)
server = app.server
app.title = "Curvature Calculator"
app.layout = LAYOUT


    #return '{"name": "Graph", "M": {"name": "np.array", "data": []}, "DM": {"name": "np.array", "data": []}, "DG": {"name": "nx.Graph", "data": {"directed": true, "multigraph": false, "graph": [], "nodes": [], "adjacency": []}}, "G": {"name": "nx.Graph", "data": {"directed": false, "multigraph": false, "graph": [], "nodes": [], "adjacency": []}}, "pos": {}, "dpos": {}, "nodes": [], "dark": false, "selected": []}'



# all changes that affect the graph directly
@app.callback(
    [Output('graph-store', 'data'),
     Output('my-graph', 'figure'),
     Output('initial', 'data'),
     Output('last-click', 'data')
    ],
    [Input('my-graph', 'selectedData'),
    Input('my-graph', 'clickData'),
    Input('layout-type', 'value'),
    Input('colortheme', 'value'),
    Input('matrix-input', 'value'),
    Input('curvature-type', 'value'),
    Input('edge-button', 'n_clicks'),
    Input('edge-weight', 'value'),
    Input('edge-delete-button', 'n_clicks'),
    Input('vertex-button', 'n_clicks'),
    Input('vertex-delete-button', 'n_clicks'),
    Input('clear-all-button', 'children'),
    Input('upload-data', 'contents'),
    Input('weighted-mode', 'value'),
    Input('idleness-store', 'data')
    ],
    [
    State('graph-store', 'data'), 
    State('last-click', 'data'),
    State('my-graph', 'figure'),
    State('upload-data', 'filename'),
    State('initial', 'data')
    ]
)
def modify_graph(clickData,
                lastClickData,
                layout_type,
                color,
                matrix,
                curvature_type,
                n_clicks,
                edge_weight, 
                n_clicks_delete,
                n_clicks_v,
                n_clicks_v_delete,
                n_clicks_clear,
                contents,
                weighted_mode,
                idleness,
                graph_state,
                last_click,
                figure, 
                filename,
                not_initial
                ):
    
    lc = LastClick()
    try:
        lc.deserialize(last_click)
    except:
        pass
    
    if not not_initial:
        G = Graph()
        my_graph_output = G.draw(layout=layout_type,
                          curvature_type=None,
                          fixed_pos=True,
                          weighted=False)
        return [json.dumps(G, cls=MyJSONEncoder),
                my_graph_output,
                'not initial',
                lc.serialize()]
    graph_store_output = graph_state
    my_graph_output = None
    #is_fixed = (is_fixed_output == 'True')
    
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    weighted = (weighted_mode == 'weighted')
    
    try:
        G = json.loads(graph_state, object_hook=MyJSONDecode)
    except:
        G = Graph()

        
    if 'layout-type' in changed_id:
        my_graph_output = G.draw(layout=layout_type,
                          curvature_type=curvature_type,
                          fixed_pos=False,
                          weighted=weighted,
                          idleness=idleness)
    
    elif 'colortheme' in changed_id:
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
        
        elif lastClickData and lastClickData['points']:
            for item in lastClickData['points']:
                curve_number = item['curveNumber']
                name = figure['data'][curve_number]['name']
                if name == 'HEATMAP':
                    G.selected = []
                    lc = LastClick(x=int(lastClickData['points'][0]['x']),
                                   y=int(lastClickData['points'][0]['y']))
                else:
                    lc = LastClick()
        
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
        lc = LastClick()
        try:
            lc.deserialize(last_click)
        except:
            pass
        if lc.x:
            G.add_node(x=lc.x, y=lc.y)
            lc.x = None
            lc.y = None
        else:
            G.add_node()
            my_graph_output = G.draw(layout=layout_type,
                              curvature_type=curvature_type,
                              fixed_pos=False,
                              weighted=weighted,
                              idleness=idleness)
            
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
        nums = []
        try:
            nums = list(map(float, matrix.replace(']', '').replace('[', '').replace(',', ' ').split()))
        except:
            pass
        N = int(np.sqrt(len(nums)))
        G.M = np.zeros(shape = (N, N), dtype = float)
        G.DM = np.zeros(shape = (N, N), dtype = float)
        if N * N != len(nums):
            pass
        else:
            for i in range(N):
                for j in range(N):
                    if nums[i * N + j] < 0.0:
                        break
                    else:
                        G.M[i][j] = nums[i * N + j]
                        G.M[j][i] = nums[i * N + j]
                        G.DM[i][j] = nums[i * N + j]
            for i in range(N):
                G.add_node(node_id=i, resize=False)
            for i in range(N):
                for j in range(N):
                    if nums[i * N + j]:
                        G.add_edge(from_node=i, to_node=j, weight=nums[i*N + j])
                

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
    graph_store_output = json.dumps(G, cls=MyJSONEncoder)
    if not my_graph_output:
        if curvature_type == 'directed':
            my_graph_output = G.draw(layout=layout_type,
                              curvature_type=curvature_type,
                              fixed_pos=True,
                              directed=True,
                              weighted=weighted,
                              idleness=idleness)
        else:
            my_graph_output = G.draw(layout=layout_type,
                                  curvature_type=curvature_type,
                                  fixed_pos=True,
                                  weighted=weighted,
                                  idleness=idleness)
    
    return [graph_store_output, my_graph_output, 'not initial', lc.serialize()]
            

@app.callback(
    Output('clear-all-button', 'children'),
    Input('confirm-delete-provider', 'submit_n_clicks'),
    prevent_initial_call=True
)
def confirm_delete(submit_n_clicks):
    if not submit_n_clicks:
        return ''
    return submit_n_clicks


@app.callback(
    [Output('matrix-input', 'value'),
     Output('edge-weight', 'value'),
     Output('idleness', 'value')],
    Input('clear-all-button', 'children'),
    prevent_initial_call=True
)
def clear_input(n_clicks):
    return ['', '', '']


@app.callback(
    [Output('warnings-area', 'value'),
     Output('idleness-store', 'data')],
    [Input('edge-weight', 'value'),
    Input('idleness', 'value'),
    Input('matrix-input', 'value')],
    State('idleness-store', 'data'),
    prevent_initial_call=True
)
def validate_and_display_warning(edge_weight, idleness_input, matrix, cur_idleness):
    idleness = cur_idleness
    warning = ''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'idleness' in changed_id:
        if idleness_input:
            try:
                idleness = float(idleness_input)
                if idleness < 0.0 or idleness > 1.0:
                    idleness = None
                    warning = 'Idleness parameter should be a real number belonging to interval [0.0, 1.0]. Computing curvature with idleness p(x)=1/(d(x)+1)...'
            except:
                idleness = None
                warning = 'Idleness parameter should be a real number belonging to interval [0.0, 1.0]. Computing curvature with idleness p(x)=1/(d(x)+1)...'
        else:
            idleness = None
            warning = 'Idleness parameter should be a real number belonging to interval [0.0, 1.0]. Computing curvature with idleness p(x)=1/(d(x)+1)...'
            
    elif 'weight' in changed_id:
        if edge_weight:
                try:
                    weight = float(edge_weight)
                    if weight < 0.0:
                        warning = 'Edge weight should be a real non-negative number. Setting the next edge weight to 1.0...'
                except:
                    pass
                
    elif 'matrix-input' in changed_id:
        nums = []
        try:
            nums = list(map(float, matrix.replace(']', '').replace('[', '').replace(',', ' ').split()))
        except:
            warning = 'Incorrect input matrix. Please check if it contains only non-negative real numbers, commas and brackets.'
    
        N = int(np.sqrt(len(nums)))
        M = np.zeros(shape=(N, N), dtype=float)
        if N * N != len(nums):
            warning = 'Incorrect input. Please check if it is a square matrix'
        elif N == 0:
            warning = 'Empty input.'
        for i in range(N):
            for j in range(N):
                if nums[i * N + j] < 0.0:
                    warning = 'Incorrect input matrix. Please check if it contains only non-negative real numbers, commas and brackets.'
                else:
                    M[i][j] = nums[i * N + j]
                    
    return [warning, idleness]


@app.callback(
    Output("download-csv", "data"),
    Input("btn_csv", "n_clicks"),
    [State('graph-store', 'data'),
     State('curvature-type', 'value'),
     State('idleness', 'value')],
    prevent_initial_call=True
)
def get_csv(n_clicks, graph, curvature, idleness):
    G = Graph()
    try:
        G = json.loads(graph, object_hook=MyJSONDecode)
    except:
        pass
    return dict(content=str(G.compute_curvatures(curvature_type=curvature, idleness=idleness)), filename="curvature.txt")


@app.callback(
    Output("download-graph", "data"),
    Input("btn_graph", "n_clicks"),
    State('graph-store', 'data'),
    prevent_initial_call=True
)
def get_graph(n_clicks, graph):
    try:
        return dict(content=graph, filename="graph.json")
    except:
        return dict(content='', filename="graph.json")


@app.callback(
    Output("download-adj", "data"),
    Input("btn_adj", "n_clicks"),
    State('graph-store', 'data'),
    prevent_initial_call=True
)
def get_graph(n_clicks, graph):
    G = Graph()
    try:
        G = json.loads(graph, object_hook=MyJSONDecode)
    except:
        pass
    return dict(content=str(G.M), filename="graph.csv")


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)