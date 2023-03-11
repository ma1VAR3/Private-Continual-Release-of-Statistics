import pickle
from dash import Dash, html, dcc
import plotly.graph_objects as go

if __name__ == '__main__':
    app = Dash(__name__)
    experiment_type1 = "best_fit"
    experiment_type2 = "quant"
    f_prefix = "./saved/" + experiment_type2 + "/" + experiment_type1 + "/"
    port = 8051 if experiment_type1 == "wrap" else 8050
    if experiment_type2 == "quant":
        port += 2
        
    with open(f_prefix + "tables.pkl", 'rb') as f:
        tables = pickle.load(f)
        
    with open(f_prefix + "figs.pkl", 'rb') as f:
        figs = pickle.load(f)
        
    with open(f_prefix + "lap_bounds.pkl", 'rb') as f:
        lap_bounds = pickle.load(f)
        
    with open(f_prefix + "epsilons.pkl", 'rb') as f:
        epsilons = pickle.load(f)
        
    with open(f_prefix + "percentiles.pkl", 'rb') as f:
        percentiles = pickle.load(f)
    
    children = [
        html.H1(children='One Shot Exponential Mechanism for Mean Estimation'),
        html.P(children='User group algo: ' + experiment_type1, className="description2"),
        html.P(children='Concentration algo: ' + experiment_type2, className="description2"),
        html.Div(children='Based on research conducted in [FS17], [Lev+21] and [GRST22]', className="description"),
    ]
    
    for f in range (len(figs)):
        lap_children = []
        for p in range(len(percentiles)):
            lap_children.append(
                html.P(children='For Laplace Mechanism, {}th Percentile MAE: '.format(percentiles[p])+str(lap_bounds[f][p]), className="laplace-bounds")
            )
        children.append(
            html.Div([
                html.P(children='For Epsilon: '+str(epsilons[f]), className="epsilon"),
                html.Div([
                        dcc.Graph(
                        id='3D-plot'+str(f),
                        figure=figs[f][0],
                        className="line-plot"
                    ),
                    dcc.Graph(
                        id='heatmap-plot'+str(f),
                        figure=figs[f][1],
                        className="heatmap-plot"
                    )
                ], className="graph-container"),
                html.Div([tables[f]], className="table-container"),
                html.Div(children=lap_children, className="lap-bounds-container")
            ], className="epsilon-container")
        )
        
    app.layout = html.Div(children=children)
        
    app.run_server(debug=False, port=port)