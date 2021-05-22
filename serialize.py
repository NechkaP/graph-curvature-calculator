#serialize.py
import json

from graph import *
from hypergraph import *

class MyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Graph):
            dct = obj.__dict__.copy()
            dct.pop('edges', None)
            dct.pop('trace_recode', None)
            return dct
        elif isinstance(obj, HyperGraph):
            dct = obj.__dict__.copy()
            dct.pop('trace_recode', None)
            return dct
        elif isinstance(obj, Node):
            dct = obj.__dict__.copy()
            dct.pop('edges_in', None)
            dct.pop('edges_out', None)
            dct.pop('trace', None)
            return dct
        elif isinstance(obj, HyperNode):
            dct = obj.__dict__.copy()
            dct.pop('trace')
            dct.pop('hyperedges')
            dct['hyperedges'] = list(obj.__dict__['hyperedges'])
            return dct
        elif isinstance(obj, HyperEdge):
            dct = obj.__dict__.copy()
            dct.pop('trace')
            dct.pop('middle_hover_trace')
            dct.pop('hypernodes')
            dct['hypernodes'] = list(obj.__dict__['hypernodes'])
            return dct
        elif isinstance(obj, nx.Graph):
            return {'name': 'nx.Graph', 'data': nx.adjacency_data(obj)}
        elif isinstance(obj, nx.DiGraph):
            return {'name': 'nx.DiGraph', 'data': nx.adjacency_data(obj)}
        elif isinstance(obj, np.ndarray):
            return {'name': 'np.array', 'data': obj.tolist()}
        elif isinstance(obj, list):
            return [MyJSONEncoder.default(self, x) for x in obj]
        return json.JSONEncoder.default(self, obj)


def MyJSONDecode(dct):
    if isinstance(dct, dict) and 'name' in dct:
        name = dct['name']
        #if name == 'Graph':
        #    print('decoding graph', dct)
        dct.pop('name', None)
        if name == 'nx.Graph':
            return nx.adjacency_graph(dct['data'], directed=False)
        elif name == 'nx.DiGraph':
            return nx.adjacency_graph(dct['data'], directed=True)
        elif name == 'np.array':
            return np.array(dct['data'])
        return getattr(modules[__name__], name)(**dct)
    return dct