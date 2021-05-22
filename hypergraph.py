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

class HyperNode:
    def __init__(self, id, x=None, y=None, **kwargs):
        self.name = self.__class__.__name__
        self.x = None if not x else x
        self.y = None if not y else y
        self.id = kwargs['id'] if 'id' in kwargs else id
        self.hyperedges = kwargs['hyperedges'] if 'hyperedges' in kwargs else set()
        self.selected = kwargs['selected'] if 'selected' in kwargs else False
        self.deleted = kwargs['deleted'] if 'deleted' in kwargs else False
        self.trace = None
    
    def draw(self, dark=False, curvature=0.0, x=None, y=None):
        if x is not None:
            self.x = x
        if y is not None:
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
        
        self.trace['name'] = int(self.id * 2)
        if self.selected:
            self.trace['marker']['color'] = color(curvature, (not dark))
            

class HyperEdge:
    def __init__(self, id, hypernodes, weight=1.0, x=None, y=None, **kwargs):
        self.x = None if not x else x
        self.y = None if not y else y
        self.name = self.__class__.__name__
        self.id = kwargs['id'] if 'id' in kwargs else id
        self.hypernodes = set(hypernodes)
        self.weight = weight
        self.trace = None
        self.middle_hover_trace = None
        self.deleted = kwargs['deleted'] if 'deleted' in kwargs else False
        self.selected = False
        
    def draw(self, dark=False, curvature=0.0, directed=False, x=None, y=None):
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        outer_size = 1.25 * min(max(10, 100 * curvature), 100)
        col = color(curvature, dark)
        if self.selected:
            col = color(curvature, not dark)
        self.trace = [go.Scatter(x=[self.x], y=[self.y],
                                text=[],
                                name='',
                                mode='markers+text',
                                textposition="bottom center",
                                hoverinfo="text",
                                hovertext='curvature=' + str(curvature),
                                marker={'size': outer_size,
                                        'color': col},
                                unselected={'marker': {'opacity': 1.0}})]
        
        self.trace.append(go.Scatter(x=[self.x], y=[self.y],
                                text=[],
                                name=self.id * 2 + 1,
                                mode='markers+text',
                                textposition="bottom center",
                                hoverinfo="text",
                                hovertext='curvature=' + str(curvature),
                                marker={'size': outer_size * 0.8,
                                        'color': plotcolor(dark)},
                                unselected={'marker': {'opacity': 1.0}}))
        
        #self.trace['name'] = int(self.id)
        #if self.selected:
        #    self.trace['marker']['color'] = color(curvature, (not dark))
        
        
class HyperGraph:
    def __init__(self, hypernodes=None, hyperedges=None, **kwargs):
        self.name = self.__class__.__name__
        
        if hypernodes is None:
            hypernodes = []
        if hyperedges is None:
            hyperedges = []
        
        if 'M' in kwargs:
            self.M = kwargs['M']
            print(self.M)
        else:
            self.M = None
            print(self.M)
        
        if 'G' in kwargs:
            self.G = kwargs['G']
        else:
            self.G = nx.Graph()
            
        self.hypernodes = hypernodes
        
        if 'pos' in kwargs:
            self.pos = kwargs['pos']
        else:
            self.pos = {}
        #else:
        #    self.pos = nx.random_layout(self.G, center=[-0.5,-0.5])#nx.bipartite_layout(self.G, nodes=[hypernode.id * 2 for hypernode in self.hypernodes])
            
        print('POS', self.pos)
        #nx.set_node_attributes(self.G, self.pos, 'pos')
        
        pos = nx.random_layout(self.G)#nx.bipartite_layout(self.G, nodes=[hypernode.id * 2 for hypernode in self.hypernodes])
        for nd in hypernodes:
            if (nd.id * 2) in self.pos.keys() or str(nd.id * 2) in self.pos.keys():
                nd.x = self.pos[str(nd.id * 2)][0] #* 100
                nd.y = self.pos[str(nd.id * 2)][1] #* 100
            else:
                if not nd.x or not nd.y:
                    nd.x = pos[nd.id * 2][0] * 100
                    nd.y = pos[nd.id * 2][1] * 100
                    if nd.x > 0.0:
                        nd.x *= -1
                    if nd.id % 2 == 0:
                        nd.y *= -1
            self.pos[str(nd.id * 2)] = (nd.x, nd.y)
        
        
        self.hyperedges = hyperedges
        for nd in hyperedges:
            if (nd.id * 2 + 1) in self.pos.keys() or str(nd.id * 2 + 1) in self.pos.keys():
                nd.x = self.pos[str(nd.id * 2 + 1)][0] #* 100
                nd.y = self.pos[str(nd.id * 2 + 1)][1] #* 100
            else:
                if not nd.x or not nd.y:
                    nd.x = pos[nd.id * 2 + 1][0] * 100
                    nd.y = pos[nd.id * 2 + 1][1] * 100
                    if nd.x < 0.0:
                        nd.x *= -1
                    if nd.id % 2 == 0:
                        nd.y *= -1
            self.pos[str(nd.id * 2 + 1)] = (nd.x, nd.y)
                
        self.trace_recode = []

        if 'dark' in kwargs:
            self.dark = kwargs['dark']
        else:
            self.dark = False
            
        if 'selected_nodes' in kwargs:
            self.selected_nodes = kwargs['selected_nodes']
        else:
            self.selected_nodes = []
            
        if 'selected_edges' in kwargs:
            self.selected_edges = kwargs['selected_edges']
        else:
            self.selected_edges = []
            
    
    def compute_curvatures(self, curvature_type=None, weighted=False, idleness=0.0):
        if len(self.hyperedges) < 1:
            return
        K = np.zeros(len(self.hyperedges))
        
        if curvature_type == 'idleness':
            K = hypergraph_curvature(self.M, idleness=idleness)
        elif curvature_type == 'forman':
            K = hypergraph_forman(self.M)
        elif curvature_type == 'directed':
            K = curvature_dir(self.DM)
        elif curvature_type == 'ollivier':
            K = hypergraph_curvature(self.M)
        return K

        
    def add_hyperedge(self, nodes_in, weight=1.0, x=None, y=None):
        print('ADD HYPEREDGE', nodes_in)
        if len(nodes_in) > 0:
            for e in self.hyperedges:
                if sorted(list(e.hypernodes)) == sorted(nodes_in):
                    return
        hyperedge = HyperEdge(len(self.hyperedges), nodes_in, weight)
        if x is not None:
            hyperedge.x = x
            hyperedge.y = y
        self.hyperedges.append(hyperedge)
        self.G.add_node(hyperedge.id * 2 + 1)
        new_M = np.zeros((len(self.hypernodes) + 1, len(self.hyperedges)))
        if self.M is not None:
            print('NEW M SHAPE', new_M.shape)
            print('SELF M SHAPE', self.M.shape)
            new_M[:,:-1] += self.M
            print(self.M)
        self.M = new_M
        print(self.M)
        self.M[0, -1] = weight
        for node_id in nodes_in:
            self.G.add_edge(node_id * 2, hyperedge.id * 2 + 1)
            self.M[int(node_id) + 1, int(hyperedge.id)] = 1 
            self.hypernodes[int(node_id)].hyperedges.append(hyperedge.id)
        
        
    def delete_hyperedge(self, id):
        hyperedge = self.hyperedges[id]
        for node_id in hyperedge.hypernodes:
            self.hypernodes[node_id].hyperedges.discard(id)
        self.M[:,id] = 0.0
        hyperedge.deleted = True
        
    
    def delete_hyperedge_by_nodes(self, nodes):
        if len(nodes) == 0:
            return False
        for e in self.hyperedges:
            if sorted(list(e.hypernodes)) == sorted(nodes):
                e.deleted = True
                iterset = set(e.hypernodes)
                for node_id in iterset:
                    self.delete_connection(node_id, e.id)
                self.M[:,e.id] = 0.0
                return True
        return False
    
        
    def add_connection(self, hypernode_id, hyperedge_id):
        self.G.add_edge(hypernode_id * 2, hyperedge_id * 2 + 1)
        self.M[int(hypernode_id) + 1, int(hyperedge_id)] = 1
        self.hypernodes[int(hypernode_id)].hyperedges.append(hyperedge_id)
        self.hyperedges[int(hyperedge_id)].hypernodes.add(hypernode_id)
        

    def delete_connection(self, hypernode_id, hyperedge_id):
        try:
            self.G.remove_edge(hypernode_id * 2, hyperedge_id * 2 + 1)
        except:
            pass
        self.M[int(hypernode_id) + 1, int(hyperedge_id)] = 0
        if hyperedge_id in self.hypernodes[int(hypernode_id)].hyperedges:
            self.hypernodes[int(hypernode_id)].hyperedges.remove(hyperedge_id)
        self.hyperedges[int(hyperedge_id)].hypernodes.discard(hypernode_id)
        
    
    def add_hypernode(self, hypernode_id=None, x=None, y=None):
        if hypernode_id is None:
            hypernode_id = len(self.hypernodes)
        if x:
            hypernode = HyperNode(id=hypernode_id, x=x, y=y)
        else:
            hypernode = HyperNode(id=hypernode_id)
        self.hypernodes.append(hypernode)
        
        new_M = np.zeros((len(self.hypernodes) + 1, len(self.hyperedges)))
        if self.M is not None:
            print('NEW M SHAPE', new_M.shape)
            print('SELF M SHAPE', self.M.shape)
            new_M[:-1] += self.M
        self.M = new_M
        print(self.M)
        self.G.add_node(hypernode_id * 2)
        
    

    def delete_hypernode(self, hypernode_id):
        hypernode = self.hypernodes[hypernode_id]
        for hyperedge_id in hypernode.hyperedges:
            hyperedge = self.hyperedges[hyperedge_id]
            hyperedge.hypernodes.discard(hypernode_id)
            if len(hyperedge.hypernodes) == 1:
                self.delete_hyperedge(hyperedge_id)
        hypernode.deleted = True
    

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
  

    def draw_connection(self, v_id, e_id):
        x0, y0 = self.hypernodes[v_id].x, self.hypernodes[v_id].y
        x1, y1 = self.hyperedges[e_id].x, self.hyperedges[e_id].y
        return go.Scatter(x=tuple([x0, x1, None]),
                                y=tuple([y0, y1, None]),
                                mode='lines',
                                name='',
                                hoverinfo='none',
                                line={'width': 3},
                                marker={'color': 'whitesmoke' if self.dark else 'silver'},
                                line_shape='spline',
                                opacity=1.0) #if curvature == 0 else min(1.0, max(0.25, curvature)))
        
#         self.middle_hover_trace = go.Scatter(x=tuple([(x0 + x1) / 2]),
#                                              y=tuple([(y0 + y1) / 2]),
#                                              hovertext=[],
#                                              mode='markers',
#                                              hoverinfo="text",
#                                              marker={'size': 3, 'color': color(curvature, dark)},
#                                              opacity=1.0 if curvature == 0 else min(1.0, max(0.25, curvature)))
        
#         #if directed or self.from_node.id > self.to_node.id:
        #self.middle_hover_trace['hovertext'] += tuple(['curvature=' + str(curvature) + ', weight=' + str(self.weight)])
        #self.middle_hover_trace['name'] = '0'
    
    def draw(self, curvature_type='ollivier', fixed_pos=True, directed=False, weighted=False, idleness=0.0):
        if idleness is None:
            idleness=0.0
        self.trace_recode_init()
        K = self.compute_curvatures(curvature_type=curvature_type, idleness=idleness)
        
        if not self.pos:
            self.pos = nx.random_layout(self.G, center=[-0.5, -0.5])#nx.bipartite_layout(self.G, nodes=[hypernode.id * 2 for hypernode in self.hypernodes])
        pos = nx.random_layout(self.G, center=[-0.5,  -0.5])#nx.bipartite_layout(self.G, nodes=[hypernode.id * 2 for hypernode in self.hypernodes])
        
        #pos = nx.bipartite_layout(self.G, nodes=[hypernode.id * 2 for hypernode in self.hypernodes])
        #self.pos = nx.spring_layout(self.G)
        #self.pos.update(nx.bipartite_layout(self.G, nodes=[hyperedge.id * 2 + 1 for hyperedge in self.hyperedges]))
        #K = self.compute_curvatures(curvature_type=curvature_type)

        for hypernode in self.hypernodes:
            if not hypernode.deleted:
                if not fixed_pos or not hypernode.x or not hypernode.y:
                    if str(hypernode.id * 2) in self.pos.keys():
                        hypernode.x = self.pos[str(hypernode.id * 2)][0] #* 100
                        hypernode.y = self.pos[str(hypernode.id * 2)][1] #* 100
                    else:
                        if hypernode.x is None:
                            hypernode.x = pos[hypernode.id * 2][0] * 100
                            hypernode.y = pos[hypernode.id * 2][1] * 100
                            if hypernode.x > 0.0:
                                hypernode.x *= -1
                            if hypernode.id % 2 == 0:
                                hypernode.y *= -1
                self.pos[str(hypernode.id * 2)] = (hypernode.x, hypernode.y)
                #if hypernode.id % 2 == 0:
                #    hypernode.y *= -1
                    
        for hyperedge in self.hyperedges:
            if not hyperedge.deleted:
                if not fixed_pos or not hyperedge.x or not hyperedge.y:
                    if str(hyperedge.id * 2 + 1) in self.pos.keys():
                        hyperedge.x = self.pos[str(hyperedge.id * 2 + 1)][0] #* 100
                        hyperedge.y = self.pos[str(hyperedge.id * 2 + 1)][1] #* 100
                    else:
                        if hyperedge.x is None:
                            hyperedge.x = pos[hyperedge.id * 2 + 1][0] * 100
                            hyperedge.y = pos[hyperedge.id * 2 + 1][1] * 100
                            if hyperedge.x < 0.0:
                                hyperedge.x *= -1
                            if hyperedge.id % 2 == 0:
                                hyperedge.y *= -1
                self.pos[str(hyperedge.id * 2 + 1)] = (hyperedge.x, hyperedge.y)
                #if hyperedge.id % 2 == 0:
                #   hyperedge.y *= -1
                    
        for v in self.hypernodes:
            if not v.deleted:
                for e in self.hyperedges:
                    if not e.deleted:
                        if self.M[v.id + 1, e.id] != 0.0:
                            self.trace_recode.append(self.draw_connection(v.id, e.id))
                
        
        for e in self.hyperedges:
            if not e.deleted:
                e.draw(dark=self.dark, curvature=K[e.id], directed=directed, x=e.x, y=e.y)#K[e.from_node.id][e.to_node.id], directed=directed)
                #self.trace_recode.append(e.trace)
                #self.trace_recode.append(e.middle_hover_trace)
                self.trace_recode += e.trace

        for hypernode in self.hypernodes:
            if not hypernode.deleted:
                if directed:
                    hypernode.draw(x=hypernode.x, y=hypernode.y, dark=self.dark, 
                                   curvature=0.0)#node_curvature(self.DM, K, n.id, weighted))
                    self.trace_recode.append(hypernode.trace)
                else:
                    hypernode.draw(x=hypernode.x, y=hypernode.y, dark=self.dark, 
                                   curvature=0.0)#node_curvature(self.M, K, hypernode.id, weighted))
                    self.trace_recode.append(hypernode.trace)

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
                                    ) for edge in []]#self.edges.values()]
                      }
        }
    
        return figure
