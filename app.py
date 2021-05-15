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

#G = Graph()
initial = True
initial2 = True

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


# all changes that affect the graph directly
@app.callback(
    [Output('graph-store', 'data'),
     Output('is-fixed-pos', 'data')
    ],
    [Input('my-graph', 'selectedData'),
    #Input('layout-type', 'value'),
    Input('colortheme', 'value'),
    Input('matrix-input', 'value'),
    #Input('curvature-type', 'value'),
    Input('edge-button', 'n_clicks'),
    Input('edge-weight', 'value'),
    Input('edge-delete-button', 'n_clicks'),
    Input('vertex-button', 'n_clicks'),
    Input('vertex-delete-button', 'n_clicks'),
    Input('clear-all-button', 'children'),
    Input('upload-data', 'contents'),
    #Input('weighted-mode', 'value'),
    #Input('last-click', 'data'),
    #Input('idleness-store', 'data')
    ],
    [
    State('graph-store', 'data'), 
    State('last-click', 'data'),
    State('my-graph', 'figure'),
    State('upload-data', 'filename'),
    State('is-fixed-pos', 'data')]
)
def modify_graph(clickData,
                #layout_type,
                color,
                matrix,
                #curvature_type,
                n_clicks,
                edge_weight, 
                n_clicks_delete,
                n_clicks_v,
                n_clicks_v_delete,
                n_clicks_clear,
                contents,
                #weighted_mode,
                graph_state,
                last_click,
                #idleness,
                figure, 
                filename,
                is_fixed=True):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
<<<<<<< HEAD
    print(changed_id, ' changed of type ', type(changed_id) )
    global initial
    if not initial:
        G = json.loads(graph_state, object_hook=MyJSONDecode)
    else:
        initial = False
        #print('no graph', graph_state)
        G = Graph()
    
    if 'colortheme' in changed_id:
        if color != None and 'D' in color:
            G.dark = True
        else:
            G.dark = False
            
    elif 'my-graph' in changed_id:
=======
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
            
    if color != None and 'D' in color:
        G.dark = True
    else:
        G.dark = False
        
    if 'layout-type' in changed_id:
        return G.draw(layout=layout_type, curvature_type=curvature_type, fixed_pos=False, weighted=weighted, idleness=idleness)
    
    #if 'colortheme' in changed_id:
    
    if 'my-graph' in changed_id:
>>>>>>> d6edb7170648e9ccba5a61c48d7c48e53a01d2c6
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
        lc = LastClick()
        print('last click =', last_click)
        try:
            lc.deserialize(last_click)
        except:
            pass
        if lc.x:
            print('adding node 1')
            print('nodes before', len(G.nodes))
            G.add_node()
            print('nodes after', len(G.nodes))
            lc.x = None
            lc.y = None
        else:
            print('adding node 2')
            print('nodes before', len(G.nodes))
            G.add_node()
            print('nodes after', len(G.nodes))

            #serialized = json.dumps(G, cls=MyJSONEncoder)
            #return [serialized, False]
            #return G.draw(layout=layout_type, curvature_type=curvature_type, fixed_pos=False, weighted=weighted, idleness=idleness)
        
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
    serialized = json.dumps(G, cls=MyJSONEncoder)
    return [serialized, is_fixed]
            

# changes that do not affect graph itself but force us to redraw it
@app.callback(
    Output('my-graph', 'figure'),
    [#Input('my-graph', 'selectedData'),
    Input('layout-type', 'value'),
    #Input('colortheme', 'value'),
    #Input('matrix-input', 'value'),
    Input('curvature-type', 'value'),
    #Input('edge-button', 'n_clicks'),
    #Input('edge-weight', 'value'),
    #Input('edge-delete-button', 'n_clicks'),
    #Input('vertex-button', 'n_clicks'),
    #Input('vertex-delete-button', 'n_clicks'),
    #Input('clear-all-button', 'children'),
    #Input('upload-data', 'contents'),
    Input('weighted-mode', 'value'),
    #Input('last-click', 'data'),
    Input('idleness-store', 'data'),
    Input('graph-store', 'data')
    ],
    [State('my-graph', 'figure'),
    #State('upload-data', 'filename')
    State('is-fixed-pos', 'data')
    ])
def change_layout(#clickData,
                  layout_type,
                  #color,
                  #matrix,
                  curvature_type,
                  #n_clicks,
                  #edge_weight, 
                  #n_clicks_delete,
                  #n_clicks_v,
                  #n_clicks_v_delete,
                  #n_clicks_clear,
                  #contents,
                  weighted_mode,
                  #last_click,
                  idleness,
                  graph_state,
                  figure, 
                  #filename,
                  is_fixed_pos
                  ):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    global initial2
    if not initial2:
        G = json.loads(graph_state, object_hook=MyJSONDecode)
    else:
        initial2 = False
        #print('no graph', graph_state)
        G = Graph()
    weighted = (weighted_mode == 'weighted')
        
    if 'layout-type' in changed_id:
        return G.draw(layout=layout_type,
                      curvature_type=curvature_type,
                      fixed_pos=False,
                      weighted=weighted,
                      idleness=idleness)  
    
    if curvature_type == 'directed':
        return G.draw(layout=layout_type,
                      curvature_type=curvature_type,
                      fixed_pos=is_fixed_pos,
                      directed=True,
                      weighted=weighted)
    
<<<<<<< HEAD
    return G.draw(layout=layout_type,
                  curvature_type=curvature_type,
                  fixed_pos=is_fixed_pos,
                  weighted=weighted,
                  idleness=idleness)

=======
    print(G.M)
    return G.draw(layout=layout_type, curvature_type=curvature_type, weighted=weighted, idleness=idleness)
>>>>>>> d6edb7170648e9ccba5a61c48d7c48e53a01d2c6

@app.callback(
    Output('clear-all-button', 'children'),
    Input('confirm-delete-provider', 'submit_n_clicks')
)
def confirm_delete(submit_n_clicks):
    if not submit_n_clicks:
        return ''
    return submit_n_clicks
    

<<<<<<< HEAD
@app.callback(
    Output('last-click', 'data'),
    Input('my-graph', 'clickData'),
    State('my-graph', 'figure')
    )
def click_processing(clickData, figure):
    if clickData is None or clickData['points'] == []:
        return ''
    curve = clickData['points'][0]['curveNumber']
    name = figure['data'][curve]['name']
    if name == 'HEATMAP':
        lc = LastClick(x=int(clickData['points'][0]['x']),
                       y=int(clickData['points'][0]['y']))
    return lc.serialize()
=======
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
>>>>>>> d6edb7170648e9ccba5a61c48d7c48e53a01d2c6


@app.callback(
    [Output('matrix-input', 'value'),
     Output('edge-weight', 'value'),
     Output('idleness', 'value')],
    Input('clear-all-button', 'children')
)
def clear_input(n_clicks):
    return ['', '', '']


@app.callback(
    [Output('warnings-area', 'value'),
     Output('idleness-store', 'data')],
    [Input('edge-weight', 'value'),
    Input('idleness', 'value'),
    Input('matrix-input', 'value')],
    State('idleness-store', 'data')
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
    
    return [warning, idleness]


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)