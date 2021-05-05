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

from Math import *

p = 0
IDLENESS = 0
N = 0
M = [[]]
K = [[]]
DARK = True
CURVATURE = 'ollivier'
WEIGHTS = 'unweighted'
THRESHOLD = 0.5
SELECTED = []
N_CLICKS = None
MATRIX = None
TRACE_RECODE = []
INITIAL = True
GRAPH = None
POS = None
ADD_VERTEX = False
DELETED = [0 for i in range(57)]
X = None
Y = None

def color(curv_value, Dark):
    if Dark:
        if curv_value < 0:
            return 'Aquamarine'
        elif curv_value == 0:
            return 'WhiteSmoke'
        else:
            return 'Tomato'
    else:
        if curv_value < 0:
            return 'LightSkyBlue'
        elif curv_value == 0:
            return 'Silver'
        else:
            return 'Crimson'
        
def plotcolor(Dark):
    if Dark:
        return 'rgb(57, 57, 57)'
    else:
        return 'WhiteSmoke'
    

class Node:
    def __init__(self, node_id='', x=None, y=None):
        #if not x:
        #    x, y = G.nodes[node_id]['pos'] * 100
        self.x = 0 if not x else x
        self.y = 0 if not y else y
        self.id = node_id
        self.deg_in = 0
        self.deg_out = 0
        self.trace = None
        self.selected = False
        self.edges_in = []
        self.edges_out = []
        self.deleted = False
    
    def draw(self, dark=False, curvature=0.0, x=None, y=None):
        if x:
            self.x = x
        if y:
            self.y = y
        self.trace = go.Scatter(x=[self.x], y=[self.y],
                                hovertext=[],
                                text=[],
                                mode='markers+text',
                                textposition="bottom center",
                                hoverinfo="none",
                                marker={'size': max(10, 100 * curvature),
                                        'color': color(curvature, dark)},
                                unselected={'marker': {'opacity': 1.0}})
        
        self.trace['name'] = self.id
        if self.selected:
            self.trace['marker']['color'] = color(curvature, (not DARK))
            

class Edge:
    def __init__(self, from_node, to_node, weight=1.0):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.trace = None
        self.middle_hover_trace = None
        
    def draw(self, dark=False, curvature=0.0):
        #weight = float(G.edges[edge]['curv']) 
        self.trace = go.Scatter(x=tuple([x0, x1, None]),
                                y=tuple([y0, y1, None]),
                                mode='lines',
                                line={'width': 3},
                                showgrid=False,
                                showaxis=False,
                                marker={'color': color(curvature, DARK)},
                                line_shape='spline',
                                opacity=1.0 if curvature == 0 else max(0.25, curvature))
        self.middle_hover_trace = go.Scatter(x=tuple([(x0 + x1) / 2]),
                                             y=tuple([(y0 + y1) / 2]),
                                             hovertext=[],
                                             mode='markers',
                                             hoverinfo="text",
                                             marker={'size': 20, 'color': color(curvature, DARK)},
                                             opacity=0)                                    
        middle_hover_trace['hovertext'] += tuple(['curvature = ' + str(curvature)])
        middle_hover_trace['name'] = ''
        edge_trace['name'] = '0'
        
        
class Graph:
    def __init__(self, nodes=[], edges=[], dark=False):
        self.M = np.zeros(shape=(len(nodes), len(nodes)))
        
        self.DG = nx.DiGraph()
        self.DG.add_nodes_from(nodes)
        self.G = nx.Graph()
        self.G.add_nodes_from(nodes)
        self.G.add_edges_from(edges)
        self.DG.add_edges_from(edges)
        
        self.pos = nx.circular_layout(self.G)
        nx.set_node_attributes(self.G, self.pos, 'pos')
        self.dpos = nx.circular_layout(self.DG)
        nx.set_node_attributes(self.DG, self.dpos, 'pos')
        
        self.nodes = []
        for nd in nodes:
            node = Node(node_id=nd, x=nd['pos'][0], y=nd['pos'][1])
            self.nodes.append(node)
        
        self.edges = edges
        for e in edges:
            from_node = self.nodes[e[0]]
            to_node = self.nodes[e[1]]
            self.add_edge(self, from_node, to_node)
            M[from_node][to_node] = 0 if not e['weight'] else e['weight']
            
        self.trace_recode = []
        self.dark = dark
        self.selected = []
        
    def update_distances():
        pass
    
    def compute_curvatures(self, curvature_type='ollivier'):
        pass
    
        
    def add_edge(self, from_node, to_node, weight=1.0, curvature=0.0):
        self.G.add_edge(from_node, to_node)
        self.DG.add_edge(from_node, to_node)
        
        from_node_ = self.nodes[from_node]
        to_node_ = self.nodes[to_node]
        edge = Edge(from_node_, to_node_) #, weight, curvature)
        from_node_.deg_out += 1
        from_node_.edges_out.append(edge)
        to_node_.deg_in += 1
        to_node_.edges_in.append(edge)
        #self.edges[(from_node, to_node)] = edge
        
    def delete_edge(self, edge):
        edge.from_node.deg_out -= 1
        edge.from_node.edges_out.remove(edge)
        edge.to_node.deg_in -= 1
        edge.to_node.edges_in.remove(edge)  
        
    def delete_edge(self, from_node, to_node):
        edge = self.edges[(from_node, to_node)]
        from_node.deg_out -= 1
        from_node.edges_out.remove(edge)
        to_node.deg_in -= 1
        to_node.edges_in.remove(edge)
    
    def add_node(self, node_id='', x=None, y=None):
        node_id = len(self.nodes)
        node = Node(node_id=node_id, x=x, y=y)
        self.nodes.append(node)
        #resize M
    
    def delete_node(self, node_id):
        node = self.nodes[node_id]
        for e in node.edges_in:
            self.delete_edge(e)
        for e in node.edges_out:
            self.delete_edge(e)
        node.deleted = True
    
    def trace_recode_init(self):
        self.trace_recode = []
        x = []
        y = []
        for i in range(-150, 150):
            x.append(i)
        for i in range(-150, 150):
            y.append(np.array(x)) 
        hm = go.Heatmap(x=x, y=x, z=np.array(y), opacity=0, showlegend=False, showscale=False)
        hm['name'] = 'HEATMAP'
        self.trace_recode.append(hm)
  
    
    def draw(self, curvature_type='ollivier', layout='circular'):
        self.trace_recode_init()
        
        self.pos = nx.circular_layout(self.G)
        if layout=='spring':
            self.pos = nx.spring_layout(self.G)
        elif layout=='random':
            self.pos = nx.random_layout(self.G)
        elif layout=='planar':
            self.pos = nx.planar_layout(self.G)
        elif layout=='shell':
            self.pos = nx.shell_layout(self.G)
        elif layout=='spectral':
            self.pos = nx.spectral_layout(self.G)

        #nx.set_node_attributes(self.G, self.pos, 'pos')
            
        #K = self.compute_curvatures(curvature_type)
        for n in self.nodes:
            if not n.deleted:
                n.x = self.pos[n.id][0] * 100
                n.y = self.pos[n.id][1] * 100
                n.draw(n.x, n.y, self.dark)
                self.trace_recode.append(n.trace)
        
        for e in self.edges:
            e.draw(self.dark)
            self.trace_recode.append(e.trace)
            

        figure = {
            "data": self.trace_recode,
            "layout": {'title': '', 
                      'showlegend': False,
                      'hovermode': 'closest', 
                      'margin': {'b': 40, 'l': 40, 'r': 40, 't': 40},
                      'xaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False},
                      'yaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False},
                      'height': 600,
                      'plot_bgcolor': plotcolor(self.dark),
                      'clickmode': 'event+select'
                      }
        }
        return figure