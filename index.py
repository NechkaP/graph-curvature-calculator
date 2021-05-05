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
Consider two very close points $x$, $y$ in a Riemannian manifold defining a tangent vector at $x$. Let $w$ be another tangent vector, and $w'$ be the tangent vector at $y$ parallel to $w$ at $x$. Following the geodesics issuing from $x$ in direction $w$ and from $y$ in direction $w'$, the geodesics will get closer in the case of positive curvature. Ricci curvature along $(xy)$ is this phenomenon, averaged in all directions $w$ at $x$. ### How can this information about graphs and hypergraphs be used? Data about graphs, which can be calculated using it, can be used in data analysis for studying network structures in the brain components. It was shown that a correlation betweenbrain diseases and curvature of the connectome graph might exist and should be examined more profoundly using machine learning methods."""

HELP_ABOUT = convert(MARKDOWN_DESCRIPTION)

LAYOUT = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Graph curvature calculator', children=[
            html.Div([
                html.Div(
                    className="row",
                    children=[
                        html.Div(
                            className="two columns",
                            children=[
                                html.Div(
                                    className="twelve columns",
                                    children=[
                                        dcc.Markdown(d("""
                                        **Curvature Type**
                                        Choose the graph curvature type. \
                                        See more information in Help.
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
                                                          'value': 'lly'},
                                                         {'label': 'Hypergraph',
                                                          'value': 'hypergraph'}
                                                     ],
                                                     value="ollivier"),
                                        html.Div(id="curv"),
                                        dcc.Upload(
                                            id='upload-data',
                                            children=html.Div([
                                                'Drag and Drop or ',
                                                html.A('Select Files')
                                            ]),
                                            style={
                                                'width': '100%',
                                                'height': '60px',
                                                'lineHeight': '60px',
                                                'borderWidth': '1px',
                                                'borderStyle': 'dashed',
                                                'borderRadius': '5px',
                                                'textAlign': 'center',
                                                'margin': '10px'
                                            },
                                            multiple=True
                                        ),
                                    ],
                                    style={'height': '300px'}
                                ),
                                html.Div(
                                    className="twelve columns",
                                    children=[
                                        dcc.Input(id='matrix-input',
                                                  type="text",
                                                  placeholder="Matrix",
                                                  debounce=True),
                                        html.Div(id='matrix')
                                    ]
                                ),
                                html.Div(
                                    className="twelve columns",
                                    children=[
                                        dcc.Checklist(id='colortheme',
                                                      options=[{'label': 'Dark color theme',
                                                                'value': 'D'}],
                                                      value=[]),
                                        html.Div(id='checkbox')
                                    ]
                                ),
                                html.Div(
                                    className="twelve columns",
                                    children=[
                                        dcc.Markdown(d("""
                                        **Graph Layout Type**
                                        Choose the layout type.
                                        See more information in Help.
                                        """)),
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
                                        html.Div(id="output")
                                    ],
                                    style={'height': '300px'}
                                )
                            ]
                        ),
                        html.Div(
                            className="eight columns",
                            children=[
                                dcc.Graph(id="my-graph", style={"height": 600})
                            ],
                            style={"height": '600px'}
                        ),
                        html.Div(
                            className="two columns",
                            children=[
                                dcc.Markdown(d("""
                                        **Hover Data**
                                        Mouse over values in the graph.
                                        """)),
                                html.Pre(id='hover-data', style=styles['pre']),
                                dcc.Markdown(d("""
                                        **Click Data**
                                        Click on points in the graph.
                                        """)),
                                html.Pre(id='click-data',
                                         style=styles['pre']),
                                html.Button('Add edge',
                                            id='edge-button'),
                                html.Button('Delete edge',
                                            id='edge-delete-button'),
                                html.Button('Add vertex',
                                            id='vertex-button'),
                                html.Button('Delete vertex',
                                            id='vertex-delete-button')
                                ])
                            ]
                        )
                    ]
                )
            ]),
        dcc.Tab(label='Help&About', children=[
            dcc.Markdown(HELP_ABOUT, dangerously_allow_html=True)
        ])
    ])

])