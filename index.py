#index.py 
import dash_core_components as dcc
import dash_html_components as html
import re
from textwrap import dedent as d
import urllib.parse
import plotly.graph_objs as go

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

def blank_figure():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None)
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False, ticklabelposition="inside right")
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False, ticklabelposition="inside top")
    
    return fig


def convert(text):
    def toimage(x):
        if x[1] and x[-2] == r'$':
            x = x[2:-2]
            img = "\n<img src='https://math.now.sh?from={}' \
            style='display: block; margin: 0.5em auto;'>\n"
            return img.format(urllib.parse.quote_plus(x))
        else:
            x = x[1:-1]
            return r'![](https://math.now.sh?from={})'.\
                   format(urllib.parse.quote_plus(x))
    return re.sub(r'\${2}([^$]+)\${2}|\$(.+?)\$',
                  lambda x: toimage(x.group()), text)


MARKDOWN_DESCRIPTION = \
    r"""## Graph and Hypergraph Curvature Calculator: Help&About
### What is curvature?
Consider two very close points $x$, $y$ in a Riemannian manifold defining a tangent vector at $x$. Let $w$ be another tangent vector, and $w'$ be the tangent vector at $y$ parallel to $w$ at $x$. Following the geodesics issuing from $x$ in direction $w$ and from $y$ in direction $w'$, the geodesics will get closer in the case of positive curvature. Ricci curvature along $(xy)$ is this phenomenon, averaged in all directions $w$ at $x$.
### How can this information about graphs and hypergraphs be used?
Data about graphs, which can be calculated using it, can be used in data analysis for studying network structures in the brain components. It was shown that a correlation betweenbrain diseases and curvature of the connectome graph might exist and should be examined more profoundly using machine learning methods."""

HELP_ABOUT = convert(MARKDOWN_DESCRIPTION)

LAYOUT = html.Div([
    dcc.Store(id='graph-store', storage_type='session'),
    dcc.Store(id='last-click', storage_type='session'),
    dcc.Store(id='idleness-store', storage_type='session'),
    dcc.Store(id='initial', storage_type='session'),
    dcc.Store(id='hypergraph-store', storage_type='session'),
    dcc.Store(id='last-click2', storage_type='session'),
    dcc.Store(id='initial2', storage_type='session'),
    dcc.Store(id='idleness-store2', storage_type='session'),
    dcc.Tabs([
        dcc.Tab(label='Graph curvature calculator', children=[
            html.Div(children=[
                html.Div(
                    className="row",
                    children=[
                        html.Div(
                            className="two columns",
                            children=[
                                html.Div(
                                    #className="twelve columns",
                                    children=[
                                        dcc.Markdown(d("""
                                        **Curvature Type**
                                        """)),
                                        dcc.Dropdown(id="curvature-type",
                                                     options=[
                                                         {'label': 'Ollivier-Ricci',
                                                          'value': 'ollivier'},
                                                         {'label': 'Ollivier-Ricci with idleness',
                                                          'value': 'idleness'},
                                                         {'label': 'Forman',
                                                          'value': 'forman'},
                                                         {'label': 'Directed',
                                                          'value': 'directed'},
                                                         {'label': 'Lin-Lu-Yau',
                                                          'value': 'lly'}
                                                     ],
                                                     value=None, 
                                                     placeholder='Choose curvature type',
                                                     style={'width':'180px', 'margin':'10px,0px,10px,0px', 'float':'left'}),
                                        html.Div(
                                            dcc.Input(placeholder='Set idleness', 
                                                      id='idleness', 
                                                      type='text',
                                                      debounce=True,
                                                      style={'width':'180px', 'margin':'10px,0px,10px,0px', 'float':'left'})
                                        ),
                                        html.Div('Node curvatures type:'),
                                        html.Div(children=[
                                            dcc.RadioItems(
                                                id='weighted-mode',
                                                options=[
                                                    {'label': 'scalar', 'value': 'scalar'},
                                                    {'label': 'weighted', 'value': 'weighted'},
                                                ],
                                                value='scalar'
                                            )],
                                            style = {'width':'185px', 'margin':'10px', 'float':'left'}
                                        ),
                                        html.Div(
                                            dcc.Textarea(id='warnings-area',
                                                         value='',
                                                         disabled=True,
                                            style={'height':'240px', 'width':'180px', 'margin':'5px', 'float':'left'}
                                            ),
                                        ),
                                        html.Div(children=[
                                                html.Button("Export as CSV", id="btn_csv",
                                                           style={'width':'180px', 'margin':'5px', 'float':'left'}),
                                                dcc.Download(id="download-csv")
                                            ]
                                        ),
                                        html.Div(children=[
                                                html.Button("Export graph", id="btn_graph",
                                                           style={'width':'180px', 'margin':'5px', 'float':'left'}),
                                                dcc.Download(id="download-graph")
                                            ],
                                        ),
                                        html.Div(children=[
                                                html.Button("Export adj.matrix", id="btn_adj",
                                                           style={'width':'180px', 'margin':'5px', 'float':'left'}),
                                                dcc.Download(id="download-adj")
                                            ],
                                        )
                                    ]
                                )
                            ]
                        ),
                        html.Div(
                            className="eight columns",
                            children=[
                                dcc.Graph(id="my-graph",
                                          style={"height": 600}, 
                                          config={'displayModeBar': True,
                                                  'modeBarButtonsToAdd': ['toggleHover']},
                                          figure = blank_figure())
                            ],
                            style={"height": '600px'}
                        ),
                        html.Div(
                            className="two columns",
                            children=[
                                html.Button('Add vertex',
                                            id='vertex-button',
                                            style={
                                                    'width': '185px',
                                                    'height': '40px',
                                                    'margin': '5px',
                                                    'float': 'right'
                                            }),
                                html.Button('Delete vertex',
                                            id='vertex-delete-button',
                                            style={
                                                    'width': '185px',
                                                    'height': '40px',
                                                    'margin': '5px',
                                                    'float': 'right'
                                            }),
                                html.Div(
                                    dcc.Input(placeholder='Set edge weight', id='edge-weight', type='text', debounce=True),
                                    style={
                                            'width': '180px',
                                            'height': '30px',
                                            'margin': '12px',
                                            'float': 'right'
                                    }
                                ),
                                html.Button('Add edge',
                                            id='edge-button',
                                    style={
                                            'width': '185px',
                                            'height': '40px',
                                            'margin': '5px',
                                            'float': 'right'
                                    }),
                                html.Button('Delete edge',
                                            id='edge-delete-button',
                                            style={
                                                    'width': '185px',
                                                    'height': '40px',
                                                    'margin': '5px',
                                                    'float': 'right'
                                            }),
                                html.Div(children=[
                                        dcc.ConfirmDialogProvider(
                                            children=html.Button('Clear all',
                                                                style={
                                                                'width': '185px',
                                                                'height': '40px',
                                                                'margin': '5px',
                                                                'float': 'right'
                                                                }
                                            ),
                                            id='confirm-delete-provider',
                                            message='Are you sure you want to clear the field? This action can not be undone.'
                                        ),
                                            html.Div(id='clear-all-button', hidden=True)
                                        ]),
                                html.Div(
                                    children=[
                                        #html.Div(className="two columns",
                                        #    children=[
                                            dcc.Markdown(d("""
                                            **Graph Layout Type**""")),
                                            dcc.Dropdown(id="layout-type",
                                                     options=[
                                                         {'label': 'circular',
                                                          'value': 'circular'},
                                                         {'label': 'random',
                                                          'value': 'random'},
                                                         {'label': 'spring',
                                                          'value': 'spring'},
                                                         {'label': 'planar',
                                                          'value': 'planar'},
                                                         {'label': 'spectral',
                                                          'value': 'spectral'},
                                                         {'label': 'shell',
                                                          'value': 'shell'}
                                                     ],
                                                     value="circular"),
                                            #html.Div(id="output")
                                        ],
                                        style={
                                            'width': '180px',
                                            'height': '75px',
                                            'margin': '10px',
                                            'float': 'right'
                                        }
                                ),
                                html.Div(
                                    #className="two columns",
                                    children=[
                                        dcc.Upload(
                                            id='upload-data',
                                            children=html.Div([
                                                'Drag and Drop or ',
                                                html.A('Select Files')
                                            ]),
                                            style={
                                                'width': '180px',
                                                'height': '60px',
                                                'lineHeight': '30px',
                                                'borderWidth': '1px',
                                                'borderStyle': 'dashed',
                                                'borderRadius': '5px',
                                                'textAlign': 'center',
                                                'margin': '10px',
                                                'float': 'right'
                                            },
                                            multiple=True
                                        )
                                    ]
                                ),
                                html.Div(
                                    #className="two columns",
                                    children=[
                                        dcc.Input(id='matrix-input',
                                                  type="text",
                                                  placeholder="Matrix",
                                                  debounce=True),
                                        html.Div(id='matrix')
                                    ],
                                    style={
                                        'width': '180px',
                                        'margin': '12px',
                                        'float': 'right'
                                    }
                                ),
                                html.Div(
                                    #className="two columns",
                                    children=[
                                        dcc.Checklist(id='colortheme',
                                                      options=[{'label': 'Dark color theme',
                                                                'value': 'D'}],
                                                      value=[]),
                                        html.Div(id='checkbox')
                                    ],
                                    style={
                                        'width': '185px',
                                        'height': '30px',
                                        'margin': '5px',
                                        'float': 'right'
                                    }
                                )
                                ])
                            ]
                        )
                    ]
                )
            ],
            style={'padding': '0','line-height': '7vh'},selected_style={'padding': '0','line-height': '7vh'}),
        dcc.Tab(label='Hypergraph curvature calculator', children=[
            html.Div(children=[
                html.Div(
                    className="row",
                    children=[
                        html.Div(
                            className="two columns",
                            children=[
                                html.Div(
                                    #className="twelve columns",
                                    children=[
                                        dcc.Markdown(d("""
                                        **Curvature Type**
                                        """)),
                                        dcc.Dropdown(id="curvature-type2",
                                                     options=[
                                                         {'label': 'Ollivier-Ricci',
                                                          'value': 'ollivier'},
                                                         {'label': 'Ollivier-Ricci with idleness',
                                                          'value': 'idleness'},
                                                         {'label': 'Forman',
                                                          'value': 'forman'},
                                                         
                                                         #{'label': 'Directed',
                                                         # 'value': 'directed'}
                                                     ],
                                                     value=None, 
                                                     placeholder='Choose curvature type',
                                                     style={'width':'180px', 'margin':'10px,0px,10px,0px', 'float':'left'}),
                                        #html.Div(
                                        #    style={'height':'38px','width':'180px', 'margin':'10px,0px,10px,0px', 'float':'left'}
                                        #),
                                        html.Div(
                                            dcc.Input(placeholder='Set idleness', 
                                                      id='idleness2', 
                                                      type='text',
                                                      debounce=True,
                                                      style={'width':'180px', 'margin':'10px,0px,10px,0px', 'float':'left'})
                                        ),
                                        html.Div('Node curvatures type:'),
                                        html.Div(children=[
                                            dcc.RadioItems(
                                                id='weighted-mode2',
                                                options=[
                                                    {'label': 'scalar', 'value': 'scalar'},
                                                    {'label': 'weighted', 'value': 'weighted'},
                                                ],
                                                value='scalar'
                                            )],
                                            style = {'width':'185px', 'margin':'10px', 'float':'left'}
                                        ),
                                        html.Div(
                                            dcc.Textarea(id='warnings-area2',
                                                         value='',
                                                         disabled=True,
                                            style={'height':'240px', 'width':'180px', 'margin':'5px', 'float':'left'}
                                            ),
                                        ),
                                        html.Div(children=[
                                                html.Button("Export as CSV", id="btn_csv2",
                                                           style={'width':'180px', 'margin':'5px', 'float':'left'}),
                                                dcc.Download(id="download-csv2")
                                            ]
                                        ),
                                        html.Div(children=[
                                                html.Button("Export hypergraph", id="btn_graph2",
                                                           style={'width':'180px', 'margin':'5px', 'float':'left'}),
                                                dcc.Download(id="download-graph2")
                                            ],
                                        ),
                                        html.Div(children=[
                                                html.Button("Export inc.matrix", id="btn_adj2",
                                                           style={'width':'180px', 'margin':'5px', 'float':'left'}),
                                                dcc.Download(id="download-adj2")
                                            ],
                                        )
                                    ]
                                )
                            ]
                        ),
                        html.Div(
                            className="eight columns",
                            children=[
                                dcc.Graph(id="my-hypergraph",
                                          style={"height": 600}, 
                                          config={'displayModeBar': True,
                                                  'modeBarButtonsToAdd': ['toggleHover']},
                                          figure = blank_figure())
                            ],
                            style={"height": '600px'}
                        ),
                        html.Div(
                            className="two columns",
                            children=[
                                html.Button('Add vertex',
                                            id='hypervertex-button',
                                            style={
                                                    'width': '185px',
                                                    'height': '40px',
                                                    'margin': '5px',
                                                    'float': 'right'
                                            }),
                                html.Button('Delete vertex',
                                            id='hypervertex-delete-button',
                                            style={
                                                    'width': '185px',
                                                    'height': '40px',
                                                    'margin': '5px',
                                                    'float': 'right'
                                            }),
                                html.Button('Add hyperedge',
                                            id='hyperedge-button',
                                            style={
                                                    'width': '185px',
                                                    'height': '40px',
                                                    'margin': '5px',
                                                    'float': 'right'
                                            }),
                                html.Div(
                                    dcc.Input(placeholder='Set hyperedge weight', id='hyperedge-weight', type='text', debounce=True),
                                    style={
                                            'width': '180px',
                                            'height': '30px',
                                            'margin': '12px',
                                            'float': 'right'
                                    }
                                ),
                                html.Button('Delete hyperedge',
                                            id='hyperedge-delete-button',
                                            style={
                                                    'width': '185px',
                                                    'height': '40px',
                                                    'margin': '5px',
                                                    'float': 'right'
                                            }),
                                html.Button('Add connection',
                                            id='connection-button',
                                    style={
                                            'width': '185px',
                                            'height': '40px',
                                            'margin': '5px',
                                            'float': 'right'
                                    }),
                                html.Button('Delete connection',
                                            id='connection-delete-button',
                                            style={
                                                    'width': '185px',
                                                    'height': '40px',
                                                    'margin': '5px',
                                                    'float': 'right'
                                            }),
                                html.Div(children=[
                                        dcc.ConfirmDialogProvider(
                                            children=html.Button('Clear all',
                                                                style={
                                                                'width': '185px',
                                                                'height': '40px',
                                                                'margin': '5px',
                                                                'float': 'right'
                                                                }
                                            ),
                                            id='confirm-delete-provider2',
                                            message='Are you sure you want to clear the field? This action can not be undone.'
                                        ),
                                            html.Div(id='clear-all-button2', hidden=True)
                                        ]),
#                                 html.Div(
#                                     children=[
#                                         #html.Div(className="two columns",
#                                         #    children=[
#                                             dcc.Markdown(d("""
#                                             **Graph Layout Type**""")),
#                                             dcc.Dropdown(id="layout-type2",
#                                                      options=[
#                                                          {'label': 'circular',
#                                                           'value': 'circular'},
#                                                          {'label': 'random',
#                                                           'value': 'random'},
#                                                          {'label': 'spring',
#                                                           'value': 'spring'},
#                                                          {'label': 'planar',
#                                                           'value': 'planar'},
#                                                          {'label': 'spectral',
#                                                           'value': 'spectral'},
#                                                          {'label': 'shell',
#                                                           'value': 'shell'}
#                                                      ],
#                                                      value="circular"),
#                                             #html.Div(id="output")
#                                         ],
#                                         style={
#                                             'width': '180px',
#                                             'height': '75px',
#                                             'margin': '10px',
#                                             'float': 'right'
#                                         }
#                                 ),
                                html.Div(
                                    #className="two columns",
                                    children=[
                                        dcc.Upload(
                                            id='upload-data2',
                                            children=html.Div([
                                                'Drag and Drop or ',
                                                html.A('Select Files')
                                            ]),
                                            style={
                                                'width': '180px',
                                                'height': '60px',
                                                'lineHeight': '30px',
                                                'borderWidth': '1px',
                                                'borderStyle': 'dashed',
                                                'borderRadius': '5px',
                                                'textAlign': 'center',
                                                'margin': '10px',
                                                'float': 'right'
                                            },
                                            multiple=True
                                        )
                                    ]
                                ),
                                html.Div(
                                    #className="two columns",
                                    children=[
                                        dcc.Input(id='matrix-input2',
                                                  type="text",
                                                  placeholder="Matrix",
                                                  debounce=True),
                                        html.Div(id='matrix2')
                                    ],
                                    style={
                                        'width': '180px',
                                        'margin': '12px',
                                        'float': 'right'
                                    }
                                ),
                                html.Div(
                                    #className="two columns",
                                    children=[
                                        dcc.Checklist(id='colortheme2',
                                                      options=[{'label': 'Dark color theme',
                                                                'value': 'D'}],
                                                      value=[]),
                                        html.Div(id='checkbox2')
                                    ],
                                    style={
                                        'width': '185px',
                                        'height': '30px',
                                        'margin': '5px',
                                        'float': 'right'
                                    }
                                )
                                ])
                            ]
                        )
                    ]
                )
            ],
            style={'padding': '0','line-height': '7vh'},selected_style={'padding': '0','line-height': '7vh'}),
        dcc.Tab(label='Help&About', children=[
            dcc.Markdown(HELP_ABOUT, dangerously_allow_html=True)],
            style={'padding': '0','line-height': '7vh'},selected_style={'padding': '0','line-height': '7vh'}
        )
    ],
    style={'height': '7vh'})
])