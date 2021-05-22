# graph.py 
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

from sys import stderr, modules
#
from Math import *

def color(curv_value, Dark):
    if Dark:
        if curv_value < 0:
            return 'Aquamarine'
        elif curv_value == 0:
            return 'DimGray'
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
    def __init__(self, node_id='', x=None, y=None, **kwargs):
        self.name = self.__class__.__name__
        #if not x:
        #    x, y = G.nodes[node_id]['pos'] * 100
        self.x = 0 if not x else x
        self.y = 0 if not y else y
        self.id = kwargs['id'] if 'id' in kwargs else node_id
        self.deg_in = kwargs['deg_in'] if 'deg_in' in kwargs else 0
        self.deg_out = kwargs['deg_out'] if 'deg_out' in kwargs else 0
        self.trace = None
        self.selected = kwargs['selected'] if 'selected' in kwargs else False
        self.edges_in = kwargs['edges_in'] if 'edges_in' in kwargs else []
        self.edges_out = kwargs['edges_out'] if 'edges_out' in kwargs else []
        self.deleted = kwargs['deleted'] if 'deleted' in kwargs else False
    
    def draw(self, dark=False, curvature=0.0, x=None, y=None):
        if x:
            self.x = x
        if y:
            self.y = y
        self.trace = go.Scatter(x=[self.x], y=[self.y],
                                text=[],
                                mode='markers+text',
                                textposition="bottom center",
                                hoverinfo="text",
                                hovertext='curvature=' + str(curvature),
                                marker={'size': min(max(10, 100 * curvature), 100),
                                        'color': color(curvature, dark)},
                                unselected={'marker': {'opacity': 1.0}})
        
        self.trace['name'] = int(self.id)
        if self.selected:
            self.trace['marker']['color'] = color(curvature, (not dark))
            

class Edge:
    def __init__(self, from_node, to_node, weight=1.0):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.trace = None
        self.middle_hover_trace = None
        
    def draw(self, dark=False, curvature=0.0, directed=False):
        x0, y0 = self.from_node.x, self.from_node.y
        x1, y1 = self.to_node.x, self.to_node.y
        self.trace = go.Scatter(x=tuple([x0, x1, None]),
                                y=tuple([y0, y1, None]),
                                mode='lines',
                                hoverinfo='none',
                                line={'width': 3},
                                marker={'color': color(curvature, dark)},
                                line_shape='spline',
                                opacity=1.0 if curvature == 0 else min(1.0, max(0.25, curvature)))
        
        self.middle_hover_trace = go.Scatter(x=tuple([(x0 + x1) / 2]),
                                             y=tuple([(y0 + y1) / 2]),
                                             hovertext=[],
                                             mode='markers',
                                             hoverinfo="text",
                                             marker={'size': 3, 'color': color(curvature, dark)},
                                             opacity=1.0 if curvature == 0 else min(1.0, max(0.25, curvature)))
        
        #if directed or self.from_node.id > self.to_node.id:
        self.middle_hover_trace['hovertext'] += tuple(['curvature=' + str(curvature) + ', weight=' + str(self.weight)])
        self.middle_hover_trace['name'] = '0'
        self.trace['name'] = '0'
        
        
class Graph:
    def __init__(self, nodes=None, edges=None, **kwargs):
        self.name = self.__class__.__name__
        
        if nodes is None:
            nodes = []
        if edges is None:
            edges = []
        
        if 'M' in kwargs:
            self.M = kwargs['M']
        else:
            self.M = np.zeros(shape=(len(nodes), len(nodes)))
        
        if 'DM' in kwargs:
            self.DM = kwargs['DM']
        else:
            self.DM = np.zeros(shape=(len(nodes), len(nodes)))
            
        if 'DG' in kwargs:
            self.DG = kwargs['DG']
        else:
            self.DG = nx.DiGraph()
            self.DG.add_nodes_from(nodes)
            self.DG.add_edges_from(edges)
        
        if 'G' in kwargs:
            self.G = kwargs['G']
        else:
            self.G = nx.Graph()
            self.G.add_nodes_from(nodes)
            self.G.add_edges_from(edges)
        
        if 'pos' in kwargs:
            self.pos = kwargs['pos']
        else:
            self.pos = nx.circular_layout(self.G)
        nx.set_node_attributes(self.G, self.pos, 'pos')
        nx.set_node_attributes(self.DG, self.pos, 'pos')
        
        self.nodes = nodes
        
        pos = nx.circular_layout(self.G)
        print('POSITION', self.pos)
        for nd in nodes:
            if nd.id in self.pos.keys():
                nd.x = self.pos[nd.id][0] * 100
                nd.y = self.pos[nd.id][1] * 100
            else:
                if not nd.x or not nd.y:
                    nd.x = pos[nd.id][0] * 100
                    nd.y = pos[nd.id][1] * 100
                self.pos[nd.id] = (nd.x, nd.y)
                
        self.edges = dict()
        for u in range(self.M.shape[0]):
            for v in range(self.M.shape[0]):
                if self.DM[u, v]:
                    self.add_edge(u, v, self.DM[u, v])
            
        self.trace_recode = []
        if 'dark' in kwargs:
            self.dark = kwargs['dark']
        else:
            self.dark = False
            
        if 'selected' in kwargs:
            self.selected = kwargs['selected']
        else:
            self.selected = []
        
    def update_distances():
        pass
    
    def compute_curvatures(self, curvature_type=None, weighted=False, idleness=None):
        if len(self.nodes) < 1:
            return
        K = np.zeros(shape=self.M.shape, dtype=float)
        
        if curvature_type == 'idleness':
            K = curvature_with_idleness(self.M, idleness=idleness)
        elif curvature_type == 'forman':
            K = forman(self.M)
        elif curvature_type == 'lly':
            K = curvature_lly(self.M)
        elif curvature_type == 'directed':
            K = curvature_dir(self.DM)
        elif curvature_type == 'ollivier':
            K = curvature(self.M)
        return np.round(K, 3)

        
    def add_edge(self, from_node, to_node, weight=1.0, curvature=0.0):
        self.G.add_edge(from_node, to_node)
        self.DG.add_edge(from_node, to_node)
        
        from_node_ = self.nodes[from_node]
        to_node_ = self.nodes[to_node]
        edge = Edge(from_node_, to_node_, weight)
        self.edges[from_node, to_node] = edge
        self.M[from_node, to_node] = weight
        self.M[to_node, from_node] = weight
        self.DM[from_node, to_node] = weight
        from_node_.deg_out += 1
        from_node_.edges_out.append(edge)
        to_node_.deg_in += 1
        to_node_.edges_in.append(edge)
        # if the graph is directed
        
    def delete_edge_(self, edge, mode=1):
        if mode:
            edge.from_node.deg_out -= 1
            if edge in edge.from_node.edges_out:
                edge.from_node.edges_out.remove(edge)
        else:
            edge.to_node.deg_in -= 1
            if edge in edge.to_node.edges_in:
                edge.to_node.edges_in.remove(edge)
        self.edges.pop((edge.from_node.id, edge.to_node.id), edge)
        self.M[edge.from_node.id][edge.to_node.id] = 0.0
        self.M[edge.to_node.id][edge.from_node.id] = 0.0
        self.DM[edge.from_node.id][edge.to_node.id] = 0.0
        self.DM[edge.to_node.id][edge.from_node.id] = 0.0
        
        
    def delete_edge(self, from_node, to_node):
        if (from_node, to_node) not in self.edges:
            return
        if (from_node, to_node) in self.edges:
            edge = self.edges[from_node, to_node]
            self.nodes[from_node].deg_out -= 1
            if edge in self.nodes[from_node].edges_out:
                self.nodes[from_node].edges_out.remove(edge)
            self.nodes[to_node].deg_in -= 1
            if edge in self.nodes[to_node].edges_in:
                self.nodes[to_node].edges_in.remove(edge)
            self.edges.pop((from_node, to_node), edge)
        if (to_node, from_node) in self.edges:
            edge = self.edges[to_node, from_node]
            self.nodes[to_node].deg_out -= 1
            if edge in self.nodes[to_node].edges_out:
                self.nodes[to_node].edges_out.remove(edge)
            self.nodes[from_node].deg_in -= 1
            if edge in self.nodes[from_node].edges_in:
                self.nodes[from_node].edges_in.remove(edge)
            self.edges.pop((to_node, from_node), edge)
        self.M[from_node][to_node] = 0.0
        self.M[to_node][from_node] = 0.0
        self.DM[from_node][to_node] = 0.0
        self.DM[to_node][from_node] = 0.0
    
    def add_node(self, node_id=None, x=None, y=None, resize=True):
        if node_id is None:
            node_id = len(self.nodes)
        if x:
            node = Node(node_id=node_id, x=x, y=y)
        else:
            node = Node(node_id=node_id)
        self.nodes.append(node)
        if resize:
            new_M = np.zeros((len(self.nodes), len(self.nodes)))
            new_M[:-1, :-1] += self.M
            self.M = new_M
            new_DM = np.zeros((len(self.nodes), len(self.nodes)))
            new_DM[:-1, :-1] += self.DM
            self.DM = new_DM
        self.G.add_node(node_id)
        self.DG.add_node(node_id)
    
    def delete_node(self, node_id):
        node = self.nodes[node_id]
        for e in node.edges_in:
            self.delete_edge_(e)
        for e in node.edges_out:
            self.delete_edge_(e, mode=0)
        node.deleted = True
    
    def trace_recode_init(self):
        self.trace_recode = []
        x = []
        y = []
        for i in range(-150, 150):
            x.append(i)
        for i in range(-150, 150):
            y.append(np.array(x)) 
        hm = go.Heatmap(x=x, y=x, z=np.array(y), opacity=0, showlegend=False, showscale=False, hoverinfo='none')
        hm['name'] = 'HEATMAP'
        self.trace_recode.append(hm)
  
    
    def draw(self, curvature_type='ollivier', layout='circular', fixed_pos=True, directed=False, weighted=False, idleness=None):
        self.trace_recode_init()
        
        if not directed:
            self.pos = nx.circular_layout(self.G)
            if layout=='spring':
                self.pos = nx.spring_layout(self.G)
            elif layout=='random':
                self.pos = nx.random_layout(self.G, center=[-0.5,-0.5])
                for item in self.pos.keys():
                    self.pos[item] *= 2
                    
            elif layout=='planar':
                self.pos = nx.planar_layout(self.G)
            elif layout=='shell':
                self.pos = nx.shell_layout(self.G)
            elif layout=='spectral':
                self.pos = nx.spectral_layout(self.G)
        else:
            self.dpos = nx.circular_layout(self.DG)
            if layout=='spring':
                self.dpos = nx.spring_layout(self.DG)
            elif layout=='random':
                self.dpos = nx.random_layout(self.DG, center=[-0.5,-0.5])
                for item in self.pos.keys():
                    self.pos[item] *= 2
            elif layout=='planar':
                self.dpos = nx.planar_layout(self.DG)
            elif layout=='shell':
                self.dpos = nx.shell_layout(self.DG)
            elif layout=='spectral':
                self.dpos = nx.spectral_layout(self.DG)
            
        K = self.compute_curvatures(curvature_type=curvature_type, idleness=idleness)

        for n in self.nodes:
            if not n.deleted:
                if not fixed_pos or not n.x or not n.y:
                    if directed:
                        n.x = self.dpos[n.id][0] * 100
                        n.y = self.dpos[n.id][1] * 100
                    else:
                        n.x = self.pos[n.id][0] * 100
                        n.y = self.pos[n.id][1] * 100
                else:
                    print(n.__dict__)

        
        for index, e in self.edges.items():
            e.draw(dark=self.dark, curvature=K[e.from_node.id][e.to_node.id], directed=directed)
            self.trace_recode.append(e.trace)
            self.trace_recode.append(e.middle_hover_trace)

        for n in self.nodes:
            if not n.deleted:
                if directed:
                    n.draw(x=n.x, y=n.y, dark=self.dark, curvature=node_curvature(self.DM, K, n.id, weighted))
                    self.trace_recode.append(n.trace)
                else:
                    n.draw(x=n.x, y=n.y, dark=self.dark, curvature=node_curvature(self.M, K, n.id, weighted))
                    self.trace_recode.append(n.trace)

        figure = {
            "data": self.trace_recode,
            "layout": {'title': '', 
                      'showlegend': False,
                      'hovermode': 'closest', 
                      'margin': {'b': 40, 'l': 40, 'r': 40, 't': 40},
                      'xaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False, 'tickvals':[]},
                      'yaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False, 'tickvals':[]},
                      'height': 600,
                      'plot_bgcolor': plotcolor(self.dark),
                      'clickmode': 'event+select',
                      'annotations': [dict(
                                        ax=(edge.from_node.x + edge.to_node.x) / 2,
                                        ay=(edge.from_node.y + edge.to_node.y) / 2, axref='x', ayref='y',
                                        x=(edge.to_node.x * 3 + edge.from_node.x) / 4,
                                        y=(edge.to_node.y * 3 + edge.from_node.y) / 4, xref='x', yref='y',
                                        showarrow=True,
                                        arrowhead=3,
                                        arrowsize=3,
                                        arrowwidth=1,
                                        opacity=edge.trace.opacity * int(directed),
                                        arrowcolor='white' if self.dark else 'slategray'
                                    ) for edge in self.edges.values()]
                      }
        }
    
        return figure