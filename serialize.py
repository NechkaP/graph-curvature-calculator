#serialize.py

class MyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Graph):
            dct = obj.__dict__.copy()
            dct.pop('edges', None)
            dct.pop('trace_recode', None)
            print('GRAPH ENCODE', dct)
            return dct
        elif isinstance(obj, Node):
            obj.__dict__.pop('edges_in', None)
            obj.__dict__.pop('edges_out', None)
            obj.__dict__.pop('trace', None)
            return obj.__dict__
        elif isinstance(obj, nx.Graph):
            return {'name': 'nx.Graph', 'data': nx.adjacency_data(obj)}
        elif isinstance(obj, nx.DiGraph):
            return {'name': 'nx.DiGraph', 'data': nx.adjacency_data(obj)}
        elif isinstance(obj, np.ndarray):
            return {'name': 'np.array', 'data': obj.tolist()}
        elif isinstance(obj, list):
            return [MyJSONEncoder.default(self, x) for x in obj]
        else:
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
        else:
            return getattr(modules[__name__], name)(**dct)
    return dct